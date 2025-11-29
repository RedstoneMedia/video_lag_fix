use std::ops::Range;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use clap::Parser;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use fast_image_resize::images::Image;
use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::event::OutputVideoFrame;
use image::{ColorType, DynamicImage, GrayImage};
use video_lag_fix::find::frame_compare::{compare_frames, preprocess_frame};
use video_lag_fix::{Args, VIDEO_DECODE_ARGS};

#[derive(Clone, Debug, Copy)]
struct Resolution {
    width: u32,
    height: u32,
}

impl FromStr for Resolution {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let Some((width_s, height_s)) = s.split_once('x') else {
            return Err("Resolution must be in <WIDTH>x<HEIGHT> format".into());
        };
        let width = width_s.parse::<u32>()
            .map_err(|e| e.to_string())?;
        let height = height_s.parse::<u32>()
            .map_err(|e| e.to_string())?;
        Ok(Resolution { width, height })
    }
}

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

    /// The resolution of the extracted frames. Uses internal `models_image` if not specified.
    #[arg(short, long)]
    resize_resolution: Option<Resolution>,

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
        cli_args.resize_resolution,
        &args
    );
}


fn resize_frame(frame: &mut OutputVideoFrame, resolution: Resolution) -> DynamicImage {
    let img = Image::from_slice_u8(frame.width, frame.height, &mut frame.data, PixelType::U8x3)
        .expect("Image buffer should be valid");
    let mut resizer = Resizer::new();
    let resize_options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    let mut resized = DynamicImage::new(resolution.width, resolution.height, ColorType::Rgb8);
    resizer.resize(&img, &mut resized, &resize_options).unwrap();
    resized
}


fn extract_similar(
    input: impl AsRef<Path>,
    to: impl AsRef<Path>,
    threshold_range: Range<f32>,
    seek: f32,
    duration: Option<f32>,
    resolution: Option<Resolution>,
    args: &Args
) {
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

    let mut last_preprocessed = None;
    let mut last_resized = None;
    let mut was_unique = true;
    for mut frame in frames_iter {
        let current = preprocess_frame(&frame, args);
        let mut current_resized = None;
        if let Some(res) = resolution {
            current_resized = Some(resize_frame(&mut frame, res));
        }

        if let Some(last) = last_preprocessed {
            let diff = compare_frames(&last, &current, args);
            let is_unique = threshold_range.contains(&diff.hash_distance);
            if is_unique {
                println!("Extracting frame #{}, diff: {:.05}", frame.frame_num, diff.hash_distance);
                let (last_img, current_img) = match resolution {
                    Some(_) => {
                        (last_resized.expect("Should exist"), current_resized.expect("Should exist"))
                    },
                    None => {
                        let resized_w = current.models_image.width();
                        let resized_h = current.models_image.height();
                        let current = GrayImage::from_vec(resized_w, resized_h, current.models_image.buffer().to_owned())
                            .expect("Should construct valid image");
                        let last = GrayImage::from_vec(resized_w, resized_h, last.models_image.into_vec())
                            .expect("Should construct valid image");
                        (image::DynamicImage::from(last), image::DynamicImage::from(current))
                    }
                };
                let current_path = to.join(format!("{:07}.png", frame.frame_num));
                current_img.save(Path::new(&current_path)).unwrap();
                let last_path = to.join(format!("{:07}.png", frame.frame_num - 1));
                if !last_path.exists() && was_unique {
                    last_img.save(Path::new(&last_path)).unwrap();
                }
                // Appease the borrow checker
                current_resized = Some(current_img);
            }
            was_unique = is_unique;
        };
        last_preprocessed = Some(current);
        last_resized = current_resized;
    }
}