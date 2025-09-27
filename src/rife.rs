use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::process::{Child, ChildStderr, ChildStdin, Command, ExitStatus, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use log::{debug, error, info};
use crate::find::DuplicateChain;

type OpenRIFEJobs = Arc<Mutex<HashMap<String, DuplicateChain>>>;

pub struct Rife {
    child: Child,
    stdin: ChildStdin,
    open_jobs: OpenRIFEJobs
}


pub struct DoneDuplicate {
    pub input0: String,
    pub input1: String,
    pub last_output: String,
    pub chain: DuplicateChain,
}

impl Rife {

    fn done_task(stdout: ChildStderr, open_jobs: OpenRIFEJobs, on_done: impl Fn(DoneDuplicate)) {
        let reader = BufReader::new(stdout);
        let done_regex = regex::Regex::new(r"^(?P<in0>\S+) (?P<in1>\S+) (?P<s>[01]\.\d+) -> (?P<out>.+?) done( \(q - (?P<q>\d+\.\d+)%\))?$").unwrap();
        for line in reader.lines() {
            let Ok(line) = line else {
                error!("RIFE Error {:?}", line.err().unwrap());
                continue;
            };
            match done_regex.captures(&line) {
                Some(captures) => {
                    let input0 = captures.name("in0").unwrap().as_str();
                    let input1 = captures.name("in1").unwrap().as_str();
                    let output = captures.name("out").unwrap().as_str();

                    let mut open_jobs = open_jobs.lock().unwrap();
                    let val = format!("{}-{}-{}", input0, input1, output);

                    if let Some(chain) = open_jobs.remove(&val) {
                        on_done(DoneDuplicate {
                            input0: input0.to_string(),
                            input1: input1.to_string(),
                            last_output: output.to_string(),
                            chain,
                        });
                    }
                },
                None => info!("RIFE: {}", line)
            }
        }
    }

    pub fn start(rife_path: impl AsRef<Path>, model_path: impl AsRef<Path>, on_done: impl Fn(DoneDuplicate) + Send + 'static) -> Self {
        let mut command = Command::new(rife_path.as_ref().join("build/rife-ncnn-vulkan"));
        command
            .arg("-o").arg("dummy.webp")
            .arg("-c")
            .arg("-v")
            .arg("-m").arg(model_path.as_ref().display().to_string())
            .stdin(Stdio::piped())
            .stderr(Stdio::piped());

        debug!("Running {:?}", command);
        let mut child = command.spawn().expect("RIFE should spawn");
        let stdin = child.stdin.take().unwrap();

        let open_jobs = Arc::new(Mutex::new(HashMap::new()));
        let stderr = child.stderr.take().unwrap();
        {
            let open_jobs = open_jobs.clone();
            std::thread::spawn(move || Self::done_task(stderr, open_jobs, on_done));
        }
        Self { child, stdin, open_jobs }
    }


    pub fn generate_in_betweens(&mut self, duplicate_chain: DuplicateChain, dir: impl AsRef<Path>) {
        let dir = dir.as_ref();

        let in_path = duplicate_chain.frames_dir.join("0.webp");
        let next_path = duplicate_chain.frames_dir.join("1.webp"); // SHOULD NOT be generated -> Would cause a repeat

        let n = duplicate_chain.frames.len() - 1;
        let mut duplicate_chain = Some(duplicate_chain);
        for i in 1..=n {
            let s = i as f32 / (n as f32 + 1.0);
            let p = get_intermediate_path(dir, i);
            writeln!(&mut self.stdin, "{},{},{},{}", in_path.display(), next_path.display(), p.display(), s).unwrap();
            // Only save the last job, so the inputs only get deleted, when really done
            if i == n {
                let mut open_jobs = self.open_jobs.lock().unwrap();
                let key = format!("{}-{}-{}", in_path.display(), next_path.display(), p.display());
                let duplicate_chain = duplicate_chain.take().expect("Can never fail, only taken once");
                open_jobs.insert(key, duplicate_chain);
            }
        }
    }

    pub fn complete(mut self) -> ExitStatus {
        drop(self.stdin);
        self.child.wait().unwrap()
    }

}


pub fn get_intermediate_path(dir: impl AsRef<Path>, patch_i: usize) -> PathBuf {
    dir.as_ref().join(format!("{:03}.png", patch_i))
}