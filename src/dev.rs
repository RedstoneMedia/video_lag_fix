use std::iter;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc;
use rand::Rng;
use crate::patch::{patch_video, Patch, PatchArgs};
use crate::utils;

pub fn insert_lag(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    n_inserts: usize,
    max_length: usize,
    patch_args: &PatchArgs,
) {
    let video_params = utils::get_video_params(input.as_ref());
    let n_frames = (video_params.framerate * video_params.duration.as_secs_f64()).floor() as usize;

    // Select insert locations
    let mut rng = rand::rng();
    let mut chains = Vec::with_capacity(n_inserts);
    'outer: while chains.len() < n_inserts {
        let new_start = rng.random_range(1..(n_frames - max_length));
        let new_length = rng.random_range(1..=max_length);// Find frames
        let new_end = new_start + new_length;
        for (start, length) in &chains {
            let end = start + length;
            // +1 because we always want one clean frame between inserts
            if *start <= new_end + 1 && new_start <= end + 1 {
                continue 'outer;
            }
        }

        chains.push((new_start, new_length));
    }
    chains.sort_unstable_by_key(|(start, _)| *start);
    // Select frames to duplicate
    println!("{:?}", chains);
    let select_dir = Path::new("tmp/selected");
    select_frames(&input, chains.iter().map(|(start, _)| *start - 1), select_dir);
    // Patch
    let (sender, receiver) = mpsc::channel::<Patch>();
    std::thread::spawn(move || {
        for (start, length) in chains {
            let frame_path = select_dir.join(format!("{:05}.png", start - 1));
            sender.send(Patch::new(
                start as u32,
                iter::repeat_n(frame_path, length).collect()
            )).expect("Should send patch");
        }
    });
    patch_video(input, output, patch_args, receiver);
}

fn select_frames(path: impl AsRef<Path>, frames: impl IntoIterator<Item=usize>, select_dir: &Path) {
    let path = path.as_ref();
    std::fs::create_dir_all(select_dir).expect("Should be able to create directory");

    let select_vf = format!("select='{}'",
        frames.into_iter()
            .map(|n| format!("eq(n,{})", n))
            .collect::<Vec<_>>()
            .join("+")
    );
    println!("{}", select_vf);
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-hwaccel").arg("cuda");
    cmd.arg("-y");
    cmd.arg("-i").arg(path);
    cmd.arg("-vf").arg(select_vf);
    cmd.arg("-fps_mode").arg("vfr");
    cmd.arg("-frame_pts").arg("1");
    cmd.arg(format!("{}/%05d.png", select_dir.display()));

    println!("Running: {:?}", cmd);
    let mut ffmpeg = cmd.spawn().expect("ffmpeg should spawn");
    let status = ffmpeg.wait().expect("Should wait for ffmpeg");
    if !status.success() {
        panic!("ffmpeg exited with {}", status);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_lag() {
        let patch_args = PatchArgs::new(Some("cuda".to_string()), [
            "-c:v", "av1_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", "36",
            "-pix_fmt", "yuv420p",
            "-an",
        ]);
        insert_lag("out/trim.mkv", "tmp/laggy.mp4", 100, 8, &patch_args);
    }
}