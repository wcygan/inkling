//! Inkling-2-Mini text front end: the Hugging Face tokenizer from `tokenizer.json` and the model's own
//! `chat_template.jinja`, rendered with minijinja and the same `tojson` / `raise_exception` helpers that
//! transformers provides.
//!
//! `encode_chat` tokenizes message text with special-token matching turned off, so a user who types
//! `<|end_message|>` gets those characters, not the control token. Template structure (role markers,
//! effort line, tool declarations) is tokenized normally. For messages without special-token strings
//! this gives exactly the IDs that transformers' `apply_chat_template(..., tokenize=True)` gives.

mod pyjson;

use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::Value as Json;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct TextModel {
    tokenizer: Tokenizer,
    literal: Tokenizer,
    env: Environment<'static>,
}

/// Private-use code points bracketing a message-text placeholder in the rendered template.
const OPEN: char = '\u{F8F0}';
const CLOSE: char = '\u{F8F1}';

/// Removes transformers' `{% generation %}` / `{% endgeneration %}` tags. They only mark assistant
/// tokens for training masks and render their body unchanged; minijinja has no custom tags.
fn strip_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(start) = rest.find("{%") {
        let Some(len) = rest[start..].find("%}") else { break };
        let tag = &rest[start + 2..start + len];
        let name = tag.trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        out.push_str(&rest[..start]);
        if name != "generation" && name != "endgeneration" {
            out.push_str(&rest[start..start + len + 2]);
        }
        rest = &rest[start + len + 2..];
    }
    out.push_str(rest);
    out
}

fn raise_exception(msg: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

impl TextModel {
    pub fn load(model_dir: &Path) -> Result<TextModel, String> {
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("{}: {e}", tok_path.display()))?;
        let mut literal = tokenizer.clone();
        literal.set_encode_special_tokens(true);
        let tpl_path = model_dir.join("chat_template.jinja");
        let source = std::fs::read_to_string(&tpl_path).map_err(|e| format!("{}: {e}", tpl_path.display()))?;
        TextModel::from_parts(tokenizer, literal, source)
    }

    fn from_parts(tokenizer: Tokenizer, literal: Tokenizer, source: String) -> Result<TextModel, String> {
        let mut env = Environment::new();
        // transformers renders with trim_blocks and lstrip_blocks.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_filter("tojson", pyjson::tojson);
        env.add_function("raise_exception", raise_exception);
        env.add_template_owned("chat", strip_generation_tags(&source))
            .map_err(|e| format!("chat_template.jinja: {e}"))?;
        Ok(TextModel { tokenizer, literal, env })
    }

    /// Renders `messages` (an OpenAI-style array) exactly as transformers' `apply_chat_template` does.
    pub fn render(&self, messages: &Json, tools: Option<&Json>, reasoning_effort: Option<&Json>) -> Result<String, String> {
        let tpl = self.env.get_template("chat").map_err(|e| e.to_string())?;
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), messages.clone());
        if let Some(t) = tools {
            ctx.insert("tools".into(), t.clone());
        }
        if let Some(e) = reasoning_effort {
            ctx.insert("reasoning_effort".into(), e.clone());
        }
        tpl.render(Value::from_serialize(&ctx)).map_err(|e| format!("chat template: {e:#}"))
    }

    /// Tokenizes text with special-token strings recognised as special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self.tokenizer.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec())
    }

    /// Tokenizes text with special-token strings treated as plain characters.
    pub fn encode_literal(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self.literal.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec())
    }

    /// Renders and tokenizes a conversation. Message text (`content`, `reasoning_content` and text parts)
    /// is tokenized literally; everything the template adds is tokenized with special tokens.
    pub fn encode_chat(&self, messages: &Json, tools: Option<&Json>, reasoning_effort: Option<&Json>) -> Result<Vec<u32>, String> {
        let mut texts: Vec<String> = Vec::new();
        let mut masked = messages.clone();
        let mut mask = |s: &mut String| {
            if s.is_empty() {
                return;
            }
            texts.push(std::mem::take(s));
            *s = format!("{OPEN}{}{CLOSE}", texts.len() - 1);
        };
        for m in masked.as_array_mut().ok_or("messages must be an array")? {
            for key in ["content", "reasoning_content"] {
                match m.get_mut(key) {
                    Some(Json::String(s)) => mask(s),
                    Some(Json::Array(parts)) => {
                        for p in parts {
                            if let Some(Json::String(s)) = p.get_mut("text") {
                                mask(s);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let rendered = self.render(&masked, tools, reasoning_effort)?;
        let mut ids = Vec::new();
        let mut rest = rendered.as_str();
        while let Some(open) = rest.find(OPEN) {
            ids.extend(self.encode(&rest[..open])?);
            let after = &rest[open + OPEN.len_utf8()..];
            let close = after.find(CLOSE).ok_or("unbalanced message placeholder")?;
            let idx: usize = after[..close].parse().map_err(|_| "bad message placeholder")?;
            ids.extend(self.encode_literal(&texts[idx])?);
            rest = &after[close + CLOSE.len_utf8()..];
        }
        ids.extend(self.encode(rest)?);
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, String> {
        self.tokenizer.decode(ids, skip_special).map_err(|e| e.to_string())
    }

    /// The ID of a special token such as `<|end_message|>`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
}
