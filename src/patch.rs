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

pub fn patch_video(args: &Args, duplicate_receiver: Receiver<DoneDuplicate>) {
    let params = VideoParams {
        framerate: 60.0,
        width: 2560,
        height: 1440,
    };

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