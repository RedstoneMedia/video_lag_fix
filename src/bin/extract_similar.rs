use std::ops::Range;
use std::path::{Path, PathBuf};
use clap::Parser;
use ffmpeg_sidecar::command::FfmpegCommand;
use image::GrayImage;
use video_lag_fix::find::frame_compare::{compare_frames, preprocess_frame};
use video_lag_fix::{Args, VIDEO_DECODE_ARGS};


/// Quick and dirty app to extract similar consecutive video frames
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// The input video file path
    #[arg(short)]
    input_path: PathBuf,
    /// The output directory of the frames
    #[arg(short)]
    output_path: PathBuf,

    /// Absolute maximum threshold for the phash, that decides which frames get exported
    #[arg(short = 'l', default_value_t = 0.0001)]
    min_threshold: f32,
    /// Absolute minimum threshold for the phash, that decides which frames get exported
    #[arg(short = 'u')]
    max_threshold: f32,

    /// The start (seconds) to seek to, from which extraction is started
    #[arg(short, long, default_value_t = 0.0)]
    seek: f32,
    /// The duration (seconds) to extract
    #[arg(short = 't', long)]
    duration: Option<f32>,

    /// Factor by which input frames are downscaled for perceptual hashing
    /// Range: 1.. (lower = higher hash resolution, more sensitive to small differences)
    #[arg(long, default_value_t = 70, verbatim_doc_comment)]
    diff_hash_resize: u32,
}

fn main() {
    let cli_args = Cli::parse();
    let args = Args {
        min_duplicate_confidence: 0.0,
        min_duplicates: 0,
        max_duplicates: 0,
        recent_motion_mean_alpha: 0.0,
        slow_motion_mean_alpha: 0.0,
        motion_compensate_threshold: 0.0,
        mul_motion_compensate_threshold_retry: 0.0,
        motion_compensate_start: 0,
        max_motion_mul: 0.0,
        diff_hash_resize: cli_args.diff_hash_resize,
        min_hash_diff: 1.0,
    };

    if !cli_args.input_path.is_file() {
        eprintln!("Error: Input file does not exist");
        std::process::exit(1);
    }

    let extract_range = cli_args.min_threshold..cli_args.max_threshold;
    extract_similar(
        cli_args.input_path,
        cli_args.output_path,
        extract_range,
        cli_args.seek,
        cli_args.duration,
        &args
    );
}

fn extract_similar(input: impl AsRef<Path>, to: impl AsRef<Path>, threshold_range: Range<f32>, seek: f32, duration: Option<f32>, args: &Args) {
    let input_path = input.as_ref();
    let to = to.as_ref();
    std::fs::create_dir_all(to).expect("Should create output directory");

    let mut command = FfmpegCommand::new();
    command.seek(seek.to_string());
    command.input(input_path.display().to_string());
    if let Some(duration) = duration {
        command.duration(duration.to_string());
    }
    command.args(VIDEO_DECODE_ARGS);

    let frames_iter = command
        .spawn().expect("Ffmpeg should spawn")
        .iter().expect("Should be able to get Ffmpeg event iterator")
        .filter_frames();

    let mut last = None;
    for frame in frames_iter {
        let current = preprocess_frame(&frame, args);
        if let Some(last) = last {
            let diff = compare_frames(&last, &current, args);
            if threshold_range.contains(&diff.hash_distance) {
                println!("Extracting frame #{}, diff: {:.05}", frame.frame_num, diff.hash_distance);
                let resized_w = current.models_image.width();
                let resized_h = current.models_image.height();
                let current = GrayImage::from_vec(resized_w, resized_h, current.models_image.buffer().to_owned())
                    .expect("Should construct valid image");
                let last = GrayImage::from_vec(resized_w, resized_h, last.models_image.into_vec())
                    .expect("Should construct valid image");

                let current_path = to.join(format!("{:07}.png", frame.frame_num));
                current.save(Path::new(&current_path)).unwrap();
                let last_path = to.join(format!("{:07}.png", frame.frame_num - 1));
                if !last_path.exists() {
                    last.save(Path::new(&last_path)).unwrap();
                }
            }
        };
        last = Some(current);
    }
}