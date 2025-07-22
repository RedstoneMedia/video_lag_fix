use std::collections::{HashSet, VecDeque};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStderr, ChildStdin, Command, ExitStatus, Stdio};
use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::event::OutputVideoFrame;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use fast_image_resize as fr;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions};
use fast_image_resize::images::{Image};
use image::{DynamicImage, ImageFormat, RgbImage, RgbaImage};
use image_hasher::HashAlg;
use rayon::prelude::*;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const DIFF_MEAN_ALPHA: f32 = 0.16;
const MUL_DUP_THRESHOLD: f32 = 0.17;
const MAX_DUP_THRESHOLD: f32 = 0.08;
const MIN_DUPLICATES: usize = 2;
const MAX_DUPLICATES: usize = 7;

const DIFF_CHUNK_SIZE: usize = 1;
const DIFF_HASH_RESIZE: u32 = 70;

const RIFE_PATH: &str = "rife-ncnn-vulkan/rife-ncnn-vulkan";
const RIFE_MODEL: &str = "models/rife-v4.26-large";

const RENDER_CQ: u8 = 28;
const RENDER_PRESET: &str = "p4";

#[inline]
fn get_luma(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32 * 77) + (g as u32 * 150) + (b as u32 * 29)) >> 8
}

/*
Naive

fn compare_frames_mae_luma(a: &OutputVideoFrame, b: &OutputVideoFrame) -> f32 {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    let n_pix = a.width * a.height;
    a.data.chunks_exact(3).zip(b.data.chunks_exact(3)).map(|(pix_a, pix_b)| {
        let y_a = get_luma(pix_a[0], pix_a[1], pix_a[2]);
        let y_b = get_luma(pix_b[0], pix_b[1], pix_b[2]);
        (y_a as i32 - y_b as i32).unsigned_abs()
    }).sum::<u32>() as f32 / n_pix as f32
}*/


fn to_grayscale(frame: &OutputVideoFrame) -> Vec<u8> {
    let src = &frame.data;
    let n = src.len() / 3;
    let mut gray = Vec::with_capacity(n);
    unsafe { gray.set_len(n) }; // SAFETY: we immediately initialize every element
    for (i, pixel) in frame.data.chunks_exact(3).enumerate() {
        let y = get_luma(pixel[0], pixel[1], pixel[2]) as u8;
        unsafe {
            *gray.get_unchecked_mut(i) = y;
        }
    }

    gray
}

fn compare_frames(a: &OutputVideoFrame, b: &OutputVideoFrame) -> f32 {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);

    let hash_width = a.width / DIFF_HASH_RESIZE;
    let hash_height = a.height / DIFF_HASH_RESIZE;
    let hasher = image_hasher::HasherConfig::new()
        .hash_size(hash_width, hash_height)
        .resize_filter(image::imageops::FilterType::Nearest) // Actual resizing happend already
        .hash_alg(HashAlg::Gradient)
        .to_hasher();

    let gray_a = to_grayscale(a);
    let gray_b = to_grayscale(b);
    let img_a = Image::from_vec_u8(a.width, a.height, gray_a, PixelType::U8).unwrap();
    let img_b = Image::from_vec_u8(b.width, b.height, gray_b, PixelType::U8).unwrap();

    let mut resizer = fr::Resizer::new();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        resizer.set_cpu_extensions(fr::CpuExtensions::Avx2);
    }
    let resize_options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));

    let width = hash_width + 1; // According to HashAlg::Gradient (row-major)
    let height = hash_height;

    let mut resized_a = DynamicImage::new(width, height, image::ColorType::L8);
    let mut resized_b = DynamicImage::new(width, height, image::ColorType::L8);

    resizer.resize(&img_a, &mut resized_a, &resize_options).unwrap();
    resizer.resize(&img_b, &mut resized_b, &resize_options).unwrap();

    let hash_a = hasher.hash_image(&resized_a);
    let hash_b = hasher.hash_image(&resized_b);
    hash_a.dist(&hash_b) as f32 / (hash_width * hash_height) as f32
}

pub struct DiffFrameIterator<I>
where
    I: Iterator<Item = OutputVideoFrame>,
{
    input_iter: I,
    chunk_size: usize,
    last_frame: Option<OutputVideoFrame>,
    buffered_diffs: VecDeque<(f32, OutputVideoFrame)>,
}


impl<I> DiffFrameIterator<I>
where
    I: Iterator<Item = OutputVideoFrame>,
{
    pub fn new(input_iter: I, chunk_size: usize) -> Self {
        Self {
            input_iter,
            chunk_size,
            last_frame: None,
            buffered_diffs: VecDeque::new(),
        }
    }

    fn load_next_chunk(&mut self) {
        // Grab chunk
        let mut chunk = Vec::with_capacity(self.chunk_size);
        for _ in 0..self.chunk_size {
            if let Some(frame) = self.input_iter.next() {
                chunk.push(frame);
            } else {
                break;
            }
        }
        if chunk.is_empty() { return; }
        // Add previous
        let mut with_previous = Vec::with_capacity(chunk.len());
        let mut prev = self.last_frame.take();
        for current in chunk.into_iter() {
            with_previous.push((prev, current.clone()));
            prev = Some(current);
        }
        self.last_frame = with_previous.last().map(|(_, f)| f.clone()); // Put new last
        // Add diff in parallel
        self.buffered_diffs = with_previous
            .into_par_iter()
            .map(|(prev_opt, current)| {
                if let Some(prev_frame) = prev_opt {
                    (compare_frames(&prev_frame, &current), current)
                } else {
                    (f32::INFINITY, current)
                }
            })
            .collect();
    }
}

impl<I> Iterator for DiffFrameIterator<I>
where
    I: Iterator<Item = OutputVideoFrame>,
{
    type Item = (f32, OutputVideoFrame);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(diff) = self.buffered_diffs.pop_front() {
            Some(diff)
        } else {
            self.load_next_chunk();
            self.buffered_diffs.pop_front()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct DuplicateChain {
    pub frames: Vec<u32>,
    pub timestamps: Vec<f32>,
    pub frames_dir: PathBuf,
}

fn frame_to_image(frame: OutputVideoFrame) -> RgbImage {
    image::ImageBuffer::from_vec(frame.width, frame.height, frame.data).unwrap()
}

fn get_duplicates(path: &str, frame_paths: impl AsRef<Path>, rife: &mut RIFE) -> Vec<DuplicateChain> {
    let iter = FfmpegCommand::new()
        .input(path)
        .rawvideo()
        .spawn().unwrap()
        .iter().unwrap();

    let mut duplicates : Vec<DuplicateChain> = Vec::new();
    let mut last_chain : Vec<OutputVideoFrame> = Vec::new();
    let frame_paths = frame_paths.as_ref();
    let s = std::time::Instant::now();
    let mut last_diff = 0.1;

    let mut diff_ema = None;

    for (diff, frame) in DiffFrameIterator::new(iter.filter_frames(), DIFF_CHUNK_SIZE) {

        // TODO: Share current timestamp with encoder thread, Publish finished duplicates

        let avg_diff = diff_ema.unwrap_or(0.0);
        let threshold = (avg_diff * MUL_DUP_THRESHOLD).min(MAX_DUP_THRESHOLD);

        //println!("#{} - #{} diff: {:.4}, ema: {:.4}, t: {:.4}", frame.frame_num - 1, frame.frame_num, diff, avg_diff, threshold);
        if diff > threshold {
            if MIN_DUPLICATES <= last_chain.len() && last_chain.len() <= MAX_DUPLICATES {
                let frames : Vec<_> = last_chain.iter().map(|f| f.frame_num).collect();
                let timestamps : Vec<_> = last_chain.iter().map(|f| f.timestamp).collect();
                println!("Duplicates found #{}-#{}, length: {}, diff_ema: {:0.4}, diff: {:.4}, at {:.3}s", frames.first().unwrap(), frames.last().unwrap(), last_chain.len(), avg_diff, last_diff, timestamps.first().unwrap());

                let start_image = frame_to_image(last_chain.remove(0));
                let end_image = frame_to_image(frame.clone());

                let dir = frame_paths.join(duplicates.len().to_string());
                fs::create_dir_all(&dir).unwrap();

                let start_frame_path = dir.join("0.webp").to_string_lossy().to_string();
                let end_frame_path = dir.join("1.webp").to_string_lossy().to_string();
                start_image.save_with_format(&start_frame_path, ImageFormat::WebP).unwrap();
                end_image.save_with_format(&end_frame_path, ImageFormat::WebP).unwrap();

                let chain = DuplicateChain {
                    frames,
                    timestamps,
                    frames_dir: dir
                };

                let patch_dir = format!("tmp/patch_{}", duplicates.len());
                fs::create_dir_all(&patch_dir).expect("Failed to create patch dir");
                rife.generate_in_betweens(
                    &chain.frames_dir,
                    &patch_dir,
                    chain.frames.len() - 1
                );
                duplicates.push(chain);
            }
            last_chain.clear();

            if diff.is_normal() {
                diff_ema = match diff_ema {
                    Some(prev) => Some(DIFF_MEAN_ALPHA * diff + (1.0 - DIFF_MEAN_ALPHA) * prev),
                    None => Some(diff),
                };
            }
        }

        if last_chain.len() > MAX_DUPLICATES {
            let mut dummy_frame = frame;
            dummy_frame.data = Vec::with_capacity(0);
            last_chain.push(dummy_frame)
        } else {
            last_chain.push(frame);
        }
        last_diff = diff;
    }
    // TODO: Last chain

    println!("Took: {}s, {}", s.elapsed().as_secs_f32(), duplicates.len());
    duplicates
}

type OpenRIFEJobs = Arc<Mutex<HashSet<String>>>;

struct RIFE {
    child: Child,
    stdin: ChildStdin,
    open_jobs: OpenRIFEJobs
}


fn try_delete(path: impl AsRef<Path>, max_tries: usize, wait: Duration) -> std::io::Result<()> {
    let mut i = 0;
    loop {
        match fs::remove_file(path.as_ref()) {
            Ok(_) => return Ok(()),
            Err(e) if i == max_tries - 1 => return Err(e),
            Err(_) => std::thread::sleep(wait),
        }
        i += 1;
    }
}

impl RIFE {

    fn done_task(stdout: ChildStderr, open_jobs: OpenRIFEJobs, on_done: impl Fn(String, String, String)) {
        let reader = BufReader::new(stdout);
        let done_regex = regex::Regex::new(r"^(?P<in0>.+) (?P<in1>.+) (?P<s>[01]\.[0-9]+) -> (?P<out>.+?) done$").unwrap();
        for line in reader.lines() {
            let Ok(line) = line else {
                println!("RIFE Error {:?}", line.err().unwrap());
                continue;
            };
            match done_regex.captures(&line) {
                Some(captures) => {
                    let input0 = captures.name("in0").unwrap().as_str();
                    let input1 = captures.name("in1").unwrap().as_str();
                    let output = captures.name("out").unwrap().as_str();
                    //let time_scale : f32 = captures.name("s").unwrap().as_str().parse().unwrap();

                    let mut open_jobs = open_jobs.lock().unwrap();
                    let val = format!("{}-{}-{}", input0, input1, output);

                    if open_jobs.contains(&val) {
                        on_done(input0.to_string(), input1.to_string(), output.to_string());
                        open_jobs.remove(&val);
                    }
                },
                None => println!("{}", line)
            }
        }
    }

    pub fn start(rife_path: &str, model_path: &str, on_done: impl Fn(String, String, String) + Send + 'static) -> Self {
        let mut command = Command::new(rife_path);
        command
            .arg("-o").arg("dummy.webp")
            .arg("-c")
            .arg("-v")
            .arg("-m").arg(model_path)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped());

        println!("{:?}", command);
        let mut child = command.spawn().unwrap();
        let stdin = child.stdin.take().unwrap();

        let open_jobs = Arc::new(Mutex::new(HashSet::new()));
        let stderr = child.stderr.take().unwrap();
        {
            let open_jobs = open_jobs.clone();
            std::thread::spawn(move || Self::done_task(stderr, open_jobs, on_done));
        }
        Self { child, stdin, open_jobs }
    }


    fn generate_in_betweens(&mut self, frames_dir: impl AsRef<Path>, dir: impl AsRef<Path>, n: usize) {
        let frames_dir = frames_dir.as_ref();
        let dir = dir.as_ref();

        let in_path = frames_dir.join("0.webp");
        let next_path = frames_dir.join("1.webp"); // SHOULD NOT be generated -> Would cause a repeat

        for i in 1..=n {
            let s = i as f32 / (n as f32 + 1.0);
            let p = dir.join(format!("{:03}.png", i));
            writeln!(&mut self.stdin, "{},{},{},{}", in_path.display(), next_path.display(), p.display(), s).unwrap();
            // Only save the last job, so the inputs only get deleted, when really done
            if i == n {
                let mut open_jobs = self.open_jobs.lock().unwrap();
                let val = format!("{}-{}-{}", in_path.display(), next_path.display(), p.display());
                open_jobs.insert(val);
            }
        }
    }

    pub fn complete(mut self) -> ExitStatus {
        drop(self.stdin);
        self.child.wait().unwrap()
    }

}

struct VideoParams {
    framerate: f64,
    width: u32,
    height: u32,
}


fn send_frames(stdin: &mut ChildStdin, duplicates: Vec<DuplicateChain>, params: &VideoParams) {
    let transparent_frame = RgbaImage::new(params.width, params.height).into_raw();
    let mut frame_counter = 0u32;
    let mut i = 0;
    let mut j_frame = 1;
    while i < duplicates.len() {
        let duplicate = &duplicates[i];
        let show_frame = duplicate.frames[j_frame];
        if show_frame != frame_counter {
            stdin.write_all(&transparent_frame).unwrap();
        } else {
            let path = format!("tmp/patch_{}/{:03}.png", i, j_frame);
            let img = image::open(path).unwrap().to_rgba8();
            let patch_frame = img.into_raw();
            stdin.write_all(&patch_frame).unwrap();
            j_frame += 1;
            if j_frame >= duplicate.frames.len() {
                i += 1;
                j_frame = 1;
                println!("Next duplicate {}", i);
            }
        }
        frame_counter += 1;
    }
}


fn main() {
    let input_path = r"trim.mkv";
    let params = VideoParams {
        framerate: 60.0,
        width: 2560,
        height: 1440,
    };
    let output_path = "out.mkv";

    let mut rife= RIFE::start(RIFE_PATH, RIFE_MODEL, move |input0, input1, _output| {
        //println!("Deleting: {}, {}", input0, input1);
        let max_tries = 3; // Sometimes RIFE still holds the files lock for some reason, even after reporting "done".
        let wait_time = Duration::from_millis(300);
        try_delete(input0, max_tries, wait_time).expect("Failed to remove input file");
        try_delete(input1, max_tries, wait_time).expect("Failed to remove input file");
    });
    let duplicates : Vec<DuplicateChain> = get_duplicates(input_path, "tmp/frames", &mut rife); // serde_json::from_str(fs::read_to_string("temp.json").unwrap().as_str()).unwrap(); //
    fs::write("temp.json", serde_json::to_string(&duplicates).unwrap()).unwrap();
    rife.complete();

    if duplicates.is_empty() {
        println!("No lag found.");
        return;
    }
    // Build ffmpeg command
    let mut cmd = Command::new("ffmpeg");
    //cmd.arg("-loglevel").arg("verbose");
    cmd.arg("-y");
    cmd.arg("-i").arg(input_path);
    cmd.arg("-f").arg("rawvideo");
    cmd.arg("-framerate").arg(params.framerate.to_string());
    cmd.arg("-pixel_format").arg("rgba");
    cmd.arg("-video_size").arg(format!("{}x{}", params.width, params.height));
    cmd.arg("-i").arg("-");
    cmd.arg("-filter_complex").arg("[0:v][1:v]overlay=0:0:eof_action=pass:format=auto");
    cmd.arg("-c:v").arg("hevc_nvenc");
    cmd.arg("-map").arg("0:a");
    cmd.arg("-c:a").arg("copy");
    cmd.arg("-rc").arg("vbr");
    cmd.arg("-cq").arg(RENDER_CQ.to_string());
    cmd.arg("-pix_fmt").arg("yuv420p");
    cmd.arg("-preset").arg(RENDER_PRESET);
    cmd.arg("-fps_mode").arg("passthrough");
    cmd.arg(output_path);
    println!("Running: {:?}", cmd);

    cmd.stdin(Stdio::piped());
    let mut ffmpeg = cmd.spawn().unwrap();
    let mut stdin = ffmpeg.stdin.take().unwrap();
    send_frames(&mut stdin, duplicates, &params);
    drop(stdin);

    let status = ffmpeg.wait().unwrap();
    println!("ffmpeg exited with: {}", status);
}
