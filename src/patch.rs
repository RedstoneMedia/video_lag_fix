use std::sync::mpsc::Receiver;
use std::process::{ChildStdin, Command, Stdio};
use image::RgbaImage;
use std::io::Write;
use std::path::Path;
use crate::Args;
use crate::rife::{get_intermediate_path, DoneDuplicate};
use crate::utils::{try_delete, try_exists, TRY_MAX_TRIES, TRY_WAIT_DURATION};

struct VideoParams {
    framerate: f64,
    width: u32,
    height: u32,
}

fn send_frames(stdin: &mut ChildStdin, duplicate_receiver: Receiver<DoneDuplicate>, params: &VideoParams) {
    let transparent_frame = RgbaImage::new(params.width, params.height).into_raw();
    let mut frame_counter = 0u32;
    let mut i = 0;
    let mut j_frame = 1;
    let mut current_duplicate = None;

    loop {
        if current_duplicate.is_none() {
            let Ok(duplicate) = duplicate_receiver.recv() else {break};
            current_duplicate = Some(duplicate);
        }
        let dup = &current_duplicate.as_ref().unwrap();

        let show_frame = dup.chain.frames[j_frame];
        if show_frame != frame_counter {
            stdin.write_all(&transparent_frame).unwrap();
        } else {
            let patch_dir = Path::new(&dup.last_output).parent().unwrap();
            let path = get_intermediate_path(patch_dir, j_frame);
            if !try_exists(&path, TRY_MAX_TRIES, TRY_WAIT_DURATION) {
                panic!("Patch frame {} does not exist", path.display());
            }
            let img = image::open(&path).unwrap().to_rgba8();
            let patch_frame = img.into_raw();
            stdin.write_all(&patch_frame).unwrap();
            j_frame += 1;
            if j_frame >= dup.chain.frames.len() {
                current_duplicate = None;
                i += 1;
                j_frame = 1;
                println!("Patching next duplicate {}", i);
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


fn get_video_params(file_path: impl AsRef<Path>) -> VideoParams {
    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=0",
            &file_path.as_ref().to_str().unwrap(),
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


pub fn patch_video(args: &Args, duplicate_receiver: Receiver<DoneDuplicate>) {
    let params = get_video_params(&args.input_path);

    // Build ffmpeg command
    let mut cmd = Command::new("ffmpeg");
    //cmd.arg("-loglevel").arg("verbose");
    cmd.arg("-y");
    cmd.arg("-hwaccel").arg("cuda");
    cmd.arg("-i").arg(args.input_path.display().to_string());
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
    cmd.arg("-cq").arg(args.render_cq.to_string());
    cmd.arg("-pix_fmt").arg("yuv420p");
    cmd.arg("-preset").arg(args.render_preset.to_string());
    cmd.arg("-fps_mode").arg("passthrough");
    cmd.arg(args.output_path.display().to_string());
    println!("Running: {:?}", cmd);

    cmd.stdin(Stdio::piped());
    let mut ffmpeg = cmd.spawn().unwrap();
    let mut stdin = ffmpeg.stdin.take().unwrap();
    send_frames(&mut stdin, duplicate_receiver, &params);
    drop(stdin);

    let status = ffmpeg.wait().unwrap();
    println!("ffmpeg exited with: {}", status);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_video_params() {
        let params = get_video_params("trimm.mkv");
        assert_eq!(params.width, 2560);
        assert_eq!(params.height, 1440);
        assert_eq!(params.framerate, 60.0);
    }

}