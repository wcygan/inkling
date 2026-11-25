//! The model dimensions the kernels hard-code, checked against `config.json` at load.

use serde_json::Value;
use std::path::Path;

/// Inkling-2-Mini text model constants. Kernels are specialised for these; `Config::load` rejects any
/// checkpoint that disagrees.
pub mod dims {
    pub const HIDDEN: usize = 2304;
    pub const LAYERS: usize = 24;
    pub const VISITS: usize = 48;
    pub const HEADS: usize = 24;
    pub const KV_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 128;
    pub const Q_DIM: usize = HEADS * HEAD_DIM; // 3072
    pub const KV_DIM: usize = KV_HEADS * HEAD_DIM; // 1024
    pub const REL_DIM: usize = 16;
    pub const R_DIM: usize = HEADS * REL_DIM; // 384
    pub const QRGS_ROWS: usize = Q_DIM + R_DIM + Q_DIM + HEADS; // 6552: q | r | gate | sink
    pub const REL_EXTENT: usize = 1024;
    pub const WINDOW: usize = 1024;
    pub const CONV_TAPS: usize = 4;
    pub const EXPERTS: usize = 128;
    pub const TOP_K: usize = 8;
    pub const INTERMEDIATE: usize = 1152;
    pub const VOCAB_PADDED: usize = 201024;
    pub const VOCAB: usize = 200058;
    pub const EOS: u32 = 200006;
    pub const LOOP_START: usize = 6;
    pub const LOOP_END: usize = 18;
    pub const LOOP_PASSES: usize = 3;
    pub const CONTROLLER_RANK: usize = 64;
    pub const BLOCK_LAYERS: usize = 6;
    pub const NVFP4_EXPERT_LAYERS: usize = 20; // layers 0..20; 20..24 are MXFP8
    pub const RMS_EPS: f32 = 1e-6;
    pub const QK_CLIP: f32 = 22.627417;
    pub const REL_CLIP: f32 = 8.0;
    pub const LOG_SCALING_FLOOR: f32 = 8192.0;
    pub const LOG_SCALING_ALPHA: f32 = 0.1;
    pub const GATE_SOFTCAP: f32 = 4.0;
    pub const UP_SOFTCAP: f32 = 25.0;
    pub const FINAL_SOFTCAP: f32 = 30.0;
    pub const ROUTE_SCALE: f32 = 8.0;
}

/// A layer is global (full-context attention) when `(layer + 1) % 6 == 0`: layers 5, 11, 17, 23.
pub const fn is_global_layer(layer: usize) -> bool {
    (layer + 1).is_multiple_of(dims::BLOCK_LAYERS)
}

#[derive(Debug, Clone)]
pub struct Config {
    pub eos_token_id: u32,
    pub model_max_length: usize,
}

fn field<'a>(v: &'a Value, path: &str) -> Result<&'a Value, String> {
    path.split('.').try_fold(v, |v, k| v.get(k).ok_or_else(|| format!("config.json: missing {path}")))
}

fn expect_usize(v: &Value, path: &str, want: usize) -> Result<(), String> {
    let got = field(v, path)?.as_u64().ok_or_else(|| format!("config.json: {path} is not an integer"))?;
    if got as usize != want {
        return Err(format!("config.json: {path} = {got}, kernels are built for {want}"));
    }
    Ok(())
}

fn expect_f64(v: &Value, path: &str, want: f64) -> Result<(), String> {
    let got = field(v, path)?.as_f64().ok_or_else(|| format!("config.json: {path} is not a number"))?;
    // Kernel constants are f32; compare at f32 precision.
    if (got - want).abs() > 1e-6 * want.abs().max(1.0) {
        return Err(format!("config.json: {path} = {got}, kernels are built for {want}"));
    }
    Ok(())
}

impl Config {
    /// Parses `config.json` and checks every dimension and constant the kernels assume.
    pub fn load(model_dir: &Path) -> Result<Config, String> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Config::from_json(&v)
    }

    pub fn from_json(v: &Value) -> Result<Config, String> {
        use dims::*;
        if field(v, "model_type")? != "inkling2_loop" {
            return Err("config.json: model_type is not inkling2_loop".into());
        }
        let t = field(v, "text_config")?;
        for (path, want) in [
            ("hidden_size", HIDDEN),
            ("num_hidden_layers", LAYERS),
            ("swa_num_attention_heads", HEADS),
            ("swa_num_key_value_heads", KV_HEADS),
            ("swa_head_dim", HEAD_DIM),
            ("num_attention_heads", HEADS),
            ("num_key_value_heads", KV_HEADS),
            ("head_dim", HEAD_DIM),
            ("d_rel", REL_DIM),
            ("rel_extent", REL_EXTENT),
            ("sliding_window_size", WINDOW),
            ("sconv_kernel_size", CONV_TAPS),
            ("n_routed_experts", EXPERTS),
            ("num_experts_per_tok", TOP_K),
            ("n_shared_experts", 0),
            ("intermediate_size", INTERMEDIATE),
            ("vocab_size", VOCAB_PADDED),
            ("unpadded_vocab_size", VOCAB),
            ("n_layers_per_attn_res_block", BLOCK_LAYERS),
            ("loop_config.core_start", LOOP_START),
            ("loop_config.core_end", LOOP_END),
            ("loop_config.num_passes", LOOP_PASSES),
            ("loop_config.gate_rank", CONTROLLER_RANK),
        ] {
            expect_usize(t, path, want)?;
        }
        for (path, want) in [
            ("rms_norm_eps", RMS_EPS as f64),
            ("qk_l2_clip_threshold", 22.627416997969522),
            ("rel_l2_clip_threshold", REL_CLIP as f64),
            ("log_scaling_n_floor", LOG_SCALING_FLOOR as f64),
            ("log_scaling_alpha", LOG_SCALING_ALPHA as f64),
            ("mlp_gate_softcap", GATE_SOFTCAP as f64),
            ("mlp_up_softcap", UP_SOFTCAP as f64),
            ("final_logit_softcapping", FINAL_SOFTCAP as f64),
            ("route_scale", ROUTE_SCALE as f64),
        ] {
            expect_f64(t, path, want)?;
        }
        for flag in ["use_gate_bias", "use_value_norm_no_gamma", "attn_res_normalize_values"] {
            if field(t, flag)?.as_bool() != Some(true) {
                return Err(format!("config.json: text_config.{flag} must be true"));
            }
        }
        if field(t, "loop_config.is_varloop")?.as_bool() != Some(false) {
            return Err("config.json: varloop models are not supported".into());
        }
        let local: Vec<u64> = field(t, "local_layer_ids")?
            .as_array()
            .ok_or("config.json: local_layer_ids is not an array")?
            .iter()
            .map(|x| x.as_u64().ok_or("config.json: local_layer_ids holds a non-integer"))
            .collect::<Result<_, _>>()?;
        let expected: Vec<u64> = (0..LAYERS as u64).filter(|&l| !is_global_layer(l as usize)).collect();
        if local != expected {
            return Err(format!("config.json: local_layer_ids {local:?} differ from the (l+1)%6 rule"));
        }
        let eos = field(v, "eos_token_id")?.as_u64().ok_or("config.json: eos_token_id")? as u32;
        if eos != EOS {
            return Err(format!("config.json: eos_token_id {eos}, expected {EOS}"));
        }
        Ok(Config {
            eos_token_id: eos,
            model_max_length: field(t, "model_max_length")?.as_u64().unwrap_or(1 << 20) as usize,
        })
    }
}
