use ffmpeg_sidecar::event::OutputVideoFrame;
use image::{DynamicImage, ImageFormat, RgbImage};
use image_hasher::HashAlg;
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, CpuExtensions};
use std::path::{Path, PathBuf};
use ffmpeg_sidecar::command::FfmpegCommand;
use std::fs;
use crate::Args;
use crate::rife::Rife;

#[inline]
fn get_luma(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32 * 77) + (g as u32 * 150) + (b as u32 * 29)) >> 8
}

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

fn compare_frames(a: &OutputVideoFrame, b: &OutputVideoFrame, args: &Args) -> f32 {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);

    let hash_width = a.width / args.diff_hash_resize;
    let hash_height = a.height / args.diff_hash_resize;
    let hasher = image_hasher::HasherConfig::new()
        .hash_size(hash_width, hash_height)
        .resize_filter(image::imageops::FilterType::Nearest) // Actual resizing happend already
        .hash_alg(HashAlg::Gradient)
        .to_hasher();

    let gray_a = to_grayscale(a);
    let gray_b = to_grayscale(b);
    let img_a = Image::from_vec_u8(a.width, a.height, gray_a, PixelType::U8).unwrap();
    let img_b = Image::from_vec_u8(b.width, b.height, gray_b, PixelType::U8).unwrap();

    let mut resizer = Resizer::new();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Avx2);
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

#[derive(Debug)]
pub struct DuplicateChain {
    pub frames: Vec<u32>,
    pub timestamps: Vec<f32>,
    pub frames_dir: PathBuf,
}

fn frame_to_image(frame: OutputVideoFrame) -> RgbImage {
    image::ImageBuffer::from_vec(frame.width, frame.height, frame.data).unwrap()
}

pub fn find_duplicates(args: &Args, rife: &mut Rife) {
    let iter = FfmpegCommand::new()
        .input(args.input_path.display().to_string())
        .rawvideo()
        .spawn().unwrap()
        .iter().unwrap();

    let s = std::time::Instant::now();

    let mut last_chain : Vec<OutputVideoFrame> = Vec::new();
    let mut last_diff = 0.1;
    let mut diff_ema = None;
    let mut i = 0;
    for frame in iter.filter_frames() {
        let diff = match last_chain.last() {
            None => f32::INFINITY,
            Some(previous) => compare_frames(&previous, &frame, &args),
        };

        let avg_diff = diff_ema.unwrap_or(0.0);
        let threshold = (avg_diff * args.mul_dup_threshold).min(args.max_dup_threshold);

        //println!("#{} - #{} diff: {:.4}, ema: {:.4}, t: {:.4}", frame.frame_num - 1, frame.frame_num, diff, avg_diff, threshold);
        if diff > threshold {
            let chain = process_new_chain(&mut last_chain, &frame, args, i);
            if let Some(chain) = chain {
                println!(
                    "Duplicates {} found #{}-#{}, length: {}, diff_ema: {:0.4}, diff: {:0.4} at {:.3}s",
                    i, chain.frames[0], chain.frames.last().unwrap(), chain.frames.len(), avg_diff, last_diff, chain.timestamps[0]
                );

                i += 1;
                let patch_dir = format!("tmp/patch_{}", i);
                fs::create_dir_all(&patch_dir).expect("Failed to create patch dir");
                rife.generate_in_betweens(
                    chain,
                    &patch_dir,
                );
            }

            if diff.is_normal() {
                diff_ema = match diff_ema {
                    Some(prev) => Some(args.diff_mean_alpha * diff + (1.0 - args.diff_mean_alpha) * prev),
                    None => Some(diff),
                };
            }
        }

        let chain_length = last_chain.len();
        if chain_length > args.max_duplicates {
            // Free up memory from too long chains using dummy frames
            for frame in last_chain.iter_mut().take(chain_length) {
                frame.data = Vec::with_capacity(0);
            }
        }
        last_chain.push(frame);
        last_diff = diff;
    }

    println!("Finding Duplicates Took: {}s", s.elapsed().as_secs_f32());
}

fn process_new_chain(last_chain: &mut Vec<OutputVideoFrame>, end_frame: &OutputVideoFrame, args: &Args, i: i32) -> Option<DuplicateChain> {
    if args.min_duplicates > last_chain.len() || last_chain.len() > args.max_duplicates {
        last_chain.clear();
        return None;
    }
    let frames: Vec<_> = last_chain.iter().map(|f| f.frame_num).collect();
    let timestamps: Vec<_> = last_chain.iter().map(|f| f.timestamp).collect();

    let start_image = frame_to_image(last_chain.remove(0));
    let end_image = frame_to_image(end_frame.clone());

    let frames_path = Path::new("tmp/frames");
    let dir = frames_path.join(i.to_string());
    fs::create_dir_all(&dir).unwrap();

    let start_frame_path = dir.join("0.webp").to_string_lossy().to_string();
    let end_frame_path = dir.join("1.webp").to_string_lossy().to_string();
    start_image.save_with_format(&start_frame_path, ImageFormat::WebP).unwrap();
    end_image.save_with_format(&end_frame_path, ImageFormat::WebP).unwrap();
    last_chain.clear();

    let chain = DuplicateChain {
        frames,
        timestamps,
        frames_dir: dir
    };
    Some(chain)
}