use std::sync::mpsc::Receiver;
use std::process::{ChildStdin, Command, Stdio};
use image::RgbaImage;
use std::io::Write;
use std::path::{Path, PathBuf};
use log::{debug, info};
use crate::rife::{get_intermediate_path, DoneDuplicate};
use crate::utils::{try_delete, try_exists, TRY_MAX_TRIES, TRY_WAIT_DURATION};

/// A sequential patch to a video.
///
/// Ranges are inclusive.
pub struct Patch {
    pub start_frame: u32,
    pub end_frame: u32,
    pub dir: PathBuf,
}

impl Patch {
    pub fn len(&self) -> u32 {
        self.end_frame - self.start_frame + 1
    }
}

impl From<DoneDuplicate> for Patch {
    fn from(value: DoneDuplicate) -> Self {
        let start_frame = value.chain.frames[1]; // 1 because 0 is not a duplicate
        let end_frame = *value.chain.frames.last().unwrap();
        let dir = Path::new(&value.last_output).parent()
            .expect("Should have parent directory").to_path_buf();
        Self {
            start_frame,
            end_frame,
            dir,
        }
    }
}

fn send_frames(stdin: &mut ChildStdin, duplicate_receiver: Receiver<Patch>, params: &VideoParams) {
    let transparent_frame = RgbaImage::new(params.width, params.height).into_raw();
    let mut frame_counter = 0u32;
    let mut patch_index = 0u32;
    let mut j_frame = 0;
    let mut current_patch = None;

    loop {
        if current_patch.is_none() {
            let Ok(duplicate) = duplicate_receiver.recv() else {break};
            assert!(duplicate.end_frame >= duplicate.start_frame);
            current_patch = Some(duplicate);
        }
        let patch = current_patch.as_ref().unwrap();

        let show_frame = patch.start_frame + j_frame;
        assert!(show_frame >= frame_counter, "{} >= {}", show_frame, frame_counter);
        if show_frame != frame_counter {
            stdin.write_all(&transparent_frame).unwrap();
        } else {
            let path = get_intermediate_path(&patch.dir, j_frame as usize + 1);
            if !try_exists(&path, TRY_MAX_TRIES, TRY_WAIT_DURATION) {
                panic!("Patch frame {} does not exist", path.display());
            }
            let img = image::open(&path).unwrap().to_rgba8();
            let patch_frame = img.into_raw();
            stdin.write_all(&patch_frame).unwrap();
            j_frame += 1;
            if j_frame >= patch.len() {
                current_patch = None;
                patch_index += 1;
                j_frame = 0;
                debug!("Patching next {}", patch_index);
            }
            // Delete the patched frame to save space
            std::thread::spawn(move || {
                std::thread::sleep(TRY_WAIT_DURATION); // Give ffmpeg time to load the frame
                try_delete(path, TRY_MAX_TRIES, TRY_WAIT_DURATION).unwrap();
            });
        }
        frame_counter += 1;
    }
}

struct VideoParams {
    framerate: f64,
    width: u32,
    height: u32,
}

fn get_video_params(file_path: impl AsRef<Path>) -> VideoParams {
    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=0",
            file_path.as_ref().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ffprobe");

    let stdout = str::from_utf8(&output.stdout).expect("Invalid UTF-8 output from ffprobe");

    let mut width = None;
    let mut height = None;
    let mut framerate = None;

    for line in stdout.lines() {
        if let Some(val) = line.strip_prefix("width=") {
            width = Some(val.parse().expect("Invalid width"));
        } else if let Some(val) = line.strip_prefix("height=") {
            height = Some(val.parse().expect("Invalid height"));
        } else if let Some(val) = line.strip_prefix("r_frame_rate=") {
            let mut parts = val.splitn(2, '/');
            let num: f64 = parts.next().unwrap_or("0").parse().expect("Invalid numerator");
            let den: f64 = parts.next().unwrap_or("1").parse().expect("Invalid denominator");
            framerate = Some(if den != 0.0 { num / den } else { 0.0 });
        }
    }

    VideoParams {
        width: width.expect("Missing width"),
        height: height.expect("Missing height"),
        framerate: framerate.expect("Missing framerate"),
    }
}

#[derive(Debug)]
pub struct PatchArgs {
    pub hw_acceleration: Option<String>,
    pub output_args: Vec<String>,
}

impl PatchArgs {

    pub fn new<I, S>(hw_acceleration: Option<String>, video_output_args: I) -> Self where
        I: IntoIterator<Item = S>,
        S: ToString
    {
        Self {
            hw_acceleration,
            output_args: video_output_args.into_iter().map(|s| s.to_string()).collect(),
        }
    }

}

pub fn patch_video(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    patch_args: &PatchArgs,
    duplicate_receiver: Receiver<Patch>
) {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();

    // Validate args
    fn deny_args(args: &[String], denied_args: &[&str]) {
        for denied in denied_args {
            if args.contains(&denied.to_string()) {
                panic!("Ffmpeg argument is now allowed to be specified: {}", denied);
            }
        }
    }
    deny_args(&patch_args.output_args, &["-i", "-f", "-filter_complex", "-fps_mode"]);

    // Build ffmpeg command
    let params = get_video_params(input_path);
    let mut cmd = Command::new("ffmpeg");
    //cmd.arg("-loglevel").arg("verbose");
    cmd.arg("-y");
    if let Some(hw_acceleration) = &patch_args.hw_acceleration {
        cmd.arg("-hwaccel").arg(hw_acceleration);
    }
    cmd.arg("-i").arg(input_path.display().to_string());
    cmd.arg("-f").arg("rawvideo");
    cmd.arg("-framerate").arg(params.framerate.to_string());
    cmd.arg("-pixel_format").arg("rgba");
    cmd.arg("-video_size").arg(format!("{}x{}", params.width, params.height));
    cmd.arg("-i").arg("-");
    cmd.arg("-filter_complex").arg("[0:v][1:v]overlay=0:0:eof_action=pass:format=auto");
    cmd.args(&patch_args.output_args);
    cmd.arg("-fps_mode").arg("passthrough");
    cmd.arg(output_path.display().to_string());
    debug!("Running: {:?}", cmd);

    cmd.stdin(Stdio::piped());
    let mut ffmpeg = cmd.spawn().unwrap();
    let mut stdin = ffmpeg.stdin.take().unwrap();
    send_frames(&mut stdin, duplicate_receiver, &params);
    drop(stdin);

    let status = ffmpeg.wait().unwrap();
    info!("ffmpeg exited with {}", status);
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use image_hasher::HashAlg;
    use super::*;

    #[test]
    fn test_get_video_params() {
        let params = get_video_params("tests/test.mkv");
        assert_eq!(params.width, 2560);
        assert_eq!(params.height, 1440);
        assert_eq!(params.framerate, 60.0);
    }

    fn split_to_frames(path: impl AsRef<Path>, frame_dir: impl AsRef<Path>) {
        let path = path.as_ref();
        let frame_dir = frame_dir.as_ref();

        if !frame_dir.exists() {
            std::fs::create_dir_all(frame_dir).expect("Should create frame test dir");
        }

        let mut ffmpeg = Command::new("ffmpeg")
            .arg("-i")
            .arg(path.display().to_string())
            .arg("-start_number").arg("0")
            .arg(format!("{}/%05d.png", frame_dir.display()))
            .spawn()
            .expect("Should spawn ffmpeg");
        let status = ffmpeg.wait().unwrap();
        if !status.success() {
            panic!("ffmpeg exited with status {}", status);
        }

        //std::fs::remove_dir_all(frame_dir).expect("Should delete frame test dir");
    }

    fn shallow_copy_dir(src: impl AsRef<Path>, dst: impl AsRef<Path>) {
        let src = src.as_ref();
        let dst = dst.as_ref();
        std::fs::create_dir_all(dst).expect("Should create dir");
        for entry in std::fs::read_dir(src).expect("Should read dir").filter_map(Result::ok) {
            if entry.file_type().expect("should get file type").is_file() {
                let file_dst = dst.join(entry.file_name());
                std::fs::copy(entry.path(), file_dst).expect("Should copy file");
            }
        }
    }

    fn assert_matching_image(a: impl AsRef<Path>, b: impl AsRef<Path>) {
        let img_a = image::open(a).expect("Should open image");
        let img_b = image::open(b).expect("Should open image");
        let hash_width = 8;
        let hash_height = 8;

        let hasher = image_hasher::HasherConfig::new()
            .hash_size(hash_width, hash_height)
            .resize_filter(image::imageops::FilterType::Lanczos3)
            .hash_alg(HashAlg::Gradient)
            .to_hasher();

        let hash_a = hasher.hash_image(&img_a);
        let hash_b = hasher.hash_image(&img_b);
        assert_eq!(hash_a, hash_b, "Image hashes do not match");
    }

    #[test]
    fn test_one_patch() {
        // Patch
        let (sender, receiver) = mpsc::channel::<Patch>();
        let patch_dir = PathBuf::from("tests/one_patch");
        let temp_patch_dir = PathBuf::from("tmp/one_patch");
        shallow_copy_dir(&patch_dir, &temp_patch_dir);
        let patch = Patch {
            start_frame: 5,
            end_frame: 8,
            dir: temp_patch_dir.clone(),
        };
        std::thread::spawn(move || {
            sender.send(patch)
        });

        let patch_args = PatchArgs::new(
            None,
            [
                "-c:v", "h264",
                "-crf", "19",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-an",
            ],
        );
        let patched_path = Path::new("tmp/one_patch.mkv");
        patch_video("tests/test.mkv", patched_path, &patch_args, receiver);
        assert!(std::fs::read_dir(&temp_patch_dir).expect("Tmp patch dir should exist").next().is_none(), "Patch dir content should be delted");

        // Check for patched
        let split_frames = Path::new("tmp/frames/one_patch");
        split_to_frames(patched_path, split_frames);
        assert_matching_image(patch_dir.join("001.png"), split_frames.join("00005.png"));
        assert_matching_image(patch_dir.join("002.png"), split_frames.join("00006.png"));
        assert_matching_image(patch_dir.join("003.png"), split_frames.join("00007.png"));
        assert_matching_image(patch_dir.join("004.png"), split_frames.join("00008.png"));

        // Cleanup
        std::fs::remove_dir_all(split_frames).expect("Should remove");
        std::fs::remove_file(patched_path).expect("Should remove");
    }

}