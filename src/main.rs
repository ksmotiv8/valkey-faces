//! valkey-faces: local face recognition with Valkey as the vector store.
//!
//! enroll    detect the largest face in an image, embed it, HSET into Valkey
//! recognize detect every face in an image, KNN-match each against the index
//! watch     live camera (or demo stream) via ffmpeg, names printed on entry/exit
//! list      who is enrolled, and with how many entries
//! forget    remove a person from the index

mod face;
mod store;
mod watch;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use face::{face_chip, load_image, Detector, Embedder};

#[derive(Parser)]
#[command(name = "valkey-faces", about = "Face recognition on Valkey vector search")]
struct Cli {
    /// Valkey connection URL
    #[arg(long, default_value = "redis://127.0.0.1:6379")]
    url: String,
    /// Directory holding the two model files
    #[arg(long, default_value = "models")]
    models: String,
    /// Cosine similarity threshold for a name match
    #[arg(long, default_value_t = 0.30)]
    threshold: f32,
    /// Minimum face size in px (must be >= 20; smaller panics the detector)
    #[arg(long, default_value_t = 20)]
    min_face: u32,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enroll the largest face in an image under a name
    Enroll { image: String, name: String },
    /// Recognize every face in an image
    Recognize { image: String },
    /// Watch a live stream and print who enters and leaves
    Watch {
        /// Use the built-in demo stream instead of a camera
        #[arg(long)]
        demo: bool,
        /// Camera device: avfoundation index on macOS ("0"), v4l2 device on Linux ("/dev/video0")
        #[arg(long, default_value_t = default_camera())]
        camera: String,
        /// Frames analyzed per second
        #[arg(long, default_value_t = 2)]
        fps: u32,
        /// Stop after N seconds (0 = run until interrupted)
        #[arg(long, default_value_t = 0)]
        duration: u32,
    },
    /// List enrolled people
    List,
    /// Remove all entries for a person
    Forget { name: String },
}

fn default_camera() -> String {
    if cfg!(target_os = "macos") { "0".into() } else { "/dev/video0".into() }
}

fn load_models(dir: &str) -> Result<(Detector, Embedder)> {
    let det_path = format!("{dir}/seeta_fd_frontal_v1.0.bin");
    let emb_path = format!("{dir}/w600k_mbf.onnx");
    let det_bytes = std::fs::read(&det_path)
        .with_context(|| format!("missing {det_path}; run ./fetch-models.sh"))?;
    let emb_bytes = std::fs::read(&emb_path)
        .with_context(|| format!("missing {emb_path}; run ./fetch-models.sh"))?;
    Ok((Detector::from_bytes(&det_bytes)?, Embedder::from_bytes(emb_bytes)))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.min_face < 20 {
        bail!("--min-face must be >= 20 (the detector's window size)");
    }
    let mut con = store::connect(&cli.url)
        .with_context(|| format!("connecting to valkey at {} (docker run -d --rm -p 6379:6379 valkey/valkey-bundle)", cli.url))?;

    match cli.cmd {
        Cmd::List => {
            let people = store::list(&mut con)?;
            if people.is_empty() {
                println!("nobody enrolled yet: valkey-faces enroll <image> <name>");
            }
            for (name, n) in people {
                println!("{name}  ({n} entr{})", if n == 1 { "y" } else { "ies" });
            }
        }
        Cmd::Forget { name } => {
            let n = store::forget(&mut con, &name)?;
            println!("removed {n} entr{} for {name}", if n == 1 { "y" } else { "ies" });
        }
        Cmd::Enroll { image, name } => {
            let (det, emb) = load_models(&cli.models)?;
            if !store::index_exists(&mut con) {
                store::create_index(&mut con)?;
                println!("created HNSW cosine index (512-D)");
            }
            let img = load_image(&image)?;
            let faces = det.detect(&img, cli.min_face);
            let Some(f) = faces.first() else { bail!("no face detected in {image}") };
            let e = emb.embed(&face_chip(&img, f))?;
            let key = store::enroll(&mut con, &name, &e)?;
            println!("enrolled {name} ({}x{} face) -> {key}", f.w, f.h);
        }
        Cmd::Recognize { image } => {
            let (det, emb) = load_models(&cli.models)?;
            let img = load_image(&image)?;
            let faces = det.detect(&img, cli.min_face);
            if faces.is_empty() {
                println!("no faces detected");
            }
            for (i, f) in faces.iter().enumerate() {
                let e = emb.embed(&face_chip(&img, f))?;
                let hit = store::knn_search(&mut con, &e, 1)?.into_iter().next();
                match hit {
                    Some(h) if 1.0 - h.dist >= cli.threshold => {
                        println!("face#{i} {}x{} @({},{}): {} (sim {:.4})", f.w, f.h, f.x, f.y, h.name, 1.0 - h.dist)
                    }
                    Some(h) => {
                        println!("face#{i} {}x{} @({},{}): stranger (nearest {} at sim {:.4})", f.w, f.h, f.x, f.y, h.name, 1.0 - h.dist)
                    }
                    None => println!("face#{i}: index is empty; enroll someone first"),
                }
            }
        }
        Cmd::Watch { demo, camera, fps, duration } => {
            let (det, emb) = load_models(&cli.models)?;
            if !store::index_exists(&mut con) {
                bail!("index is empty; enroll at least one person first");
            }
            let opts = watch::WatchOpts {
                demo,
                camera,
                fps,
                duration,
                threshold: cli.threshold,
                min_face: cli.min_face,
            };
            watch::run(&mut con, &det, &emb, &opts, "demo-faces")?;
        }
    }
    Ok(())
}
