//! Parser for the CN's admin-command script, delivered by StarRocks'
//! `ADMIN EXECUTE ON <node_id> '<script>'` (the FE forwards the script string verbatim over the
//! `execute_command` RPC; upstream BEs interpret it as a C++ debug script — this CN accepts only
//! the grammar below, never code).
//!
//! Line-oriented: each non-empty line is one command; `#`-prefixed lines are comments. Commands
//! run in order and stop at the first failure (re-runs are safe: re-pinning a name replaces the
//! entry). There is deliberately no quote layer — the string already crossed the FE's SQL-literal
//! layer — so values end at whitespace: paths, names, and column names containing spaces or `=`
//! are unsupported and rejected by construction.
//!
//! ```text
//! pin_table path=<file-or-glob> tier=gpu|host name=<name> [cols=c1,c2,...]
//!           [format=parquet|duckdb] [schema=<schema>]
//! unpin_table <name>
//! ```

use crate::fragment_executor::{PinTableSpec, PinTier};

/// One parsed admin command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdminCommand {
    /// Pin a table into the engine's scan cache.
    PinTable(PinTableSpec),
    /// Remove a pinned entry by name.
    UnpinTable {
        /// The pin-registry key to remove.
        name: String,
    },
}

/// Parses a full script into commands, or an error naming the offending line.
pub fn parse_script(script: &str) -> Result<Vec<AdminCommand>, String> {
    let mut commands = Vec::new();
    for (index, line) in script.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        commands.push(
            parse_line(line).map_err(|err| format!("line {}: {err}", index + 1))?,
        );
    }
    if commands.is_empty() {
        return Err(format!("empty script; supported commands: {GRAMMAR}"));
    }
    Ok(commands)
}

const GRAMMAR: &str = "pin_table path=<file-or-glob> tier=gpu|host name=<name> \
                       [cols=c1,c2,...] [format=parquet|duckdb] [schema=<schema>] | \
                       unpin_table <name>";

fn parse_line(line: &str) -> Result<AdminCommand, String> {
    let mut tokens = line.split_ascii_whitespace();
    let verb = tokens.next().expect("caller skips empty lines");
    match verb {
        "pin_table" => parse_pin_table(tokens),
        "unpin_table" => {
            let name = tokens
                .next()
                .ok_or_else(|| "unpin_table requires a name".to_string())?;
            if let Some(extra) = tokens.next() {
                return Err(format!("unpin_table takes one name; unexpected '{extra}'"));
            }
            Ok(AdminCommand::UnpinTable {
                name: name.to_string(),
            })
        }
        other => Err(format!("unknown command '{other}'; supported: {GRAMMAR}")),
    }
}

fn parse_pin_table<'a>(tokens: impl Iterator<Item = &'a str>) -> Result<AdminCommand, String> {
    let mut path = None;
    let mut tier = None;
    let mut name = None;
    let mut cols = None;
    let mut format = None;
    let mut schema = None;

    for token in tokens {
        let (key, value) = token
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got '{token}'"))?;
        if value.is_empty() {
            return Err(format!("empty value for '{key}'"));
        }
        let slot: &mut Option<String> = match key {
            "path" => &mut path,
            "tier" => &mut tier,
            "name" => &mut name,
            "cols" => &mut cols,
            "format" => &mut format,
            "schema" => &mut schema,
            other => return Err(format!("unknown key '{other}'; supported: {GRAMMAR}")),
        };
        if slot.is_some() {
            return Err(format!("duplicate key '{key}'"));
        }
        *slot = Some(value.to_string());
    }

    let tier = match tier.as_deref() {
        Some("gpu") => PinTier::Gpu,
        Some("host") => PinTier::Host,
        Some(other) => return Err(format!("tier must be gpu or host, got '{other}'")),
        None => return Err("pin_table requires tier=gpu|host".to_string()),
    };
    let name = name.ok_or_else(|| "pin_table requires name=<name>".to_string())?;
    match format.as_deref() {
        None | Some("parquet") | Some("duckdb") => {}
        Some(other) => return Err(format!("format must be parquet or duckdb, got '{other}'")),
    }
    if path.is_none() && format.as_deref() != Some("duckdb") {
        return Err("pin_table requires path=<file-or-glob> unless format=duckdb".to_string());
    }
    let cols = cols.map(|list| {
        list.split(',')
            .filter(|column| !column.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    if matches!(&cols, Some(columns) if columns.is_empty()) {
        return Err("cols must name at least one column".to_string());
    }

    Ok(AdminCommand::PinTable(PinTableSpec {
        path,
        tier,
        name,
        cols,
        format,
        schema,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin(script: &str) -> PinTableSpec {
        match parse_script(script).unwrap().remove(0) {
            AdminCommand::PinTable(spec) => spec,
            other => panic!("expected PinTable, got {other:?}"),
        }
    }

    #[test]
    fn parses_full_pin_table() {
        let spec = pin(
            "pin_table path=/data/lineitem/*.parquet tier=gpu name=lineitem \
             cols=l_quantity,l_shipdate format=parquet",
        );
        assert_eq!(spec.path.as_deref(), Some("/data/lineitem/*.parquet"));
        assert_eq!(spec.tier, PinTier::Gpu);
        assert_eq!(spec.name, "lineitem");
        assert_eq!(
            spec.cols,
            Some(vec!["l_quantity".to_string(), "l_shipdate".to_string()])
        );
        assert_eq!(spec.format.as_deref(), Some("parquet"));
        assert_eq!(spec.schema, None);
    }

    #[test]
    fn parses_duckdb_pin_without_path() {
        let spec = pin("pin_table tier=host name=t format=duckdb schema=main");
        assert_eq!(spec.path, None);
        assert_eq!(spec.tier, PinTier::Host);
        assert_eq!(spec.schema.as_deref(), Some("main"));
    }

    #[test]
    fn parses_unpin_and_comments_and_blank_lines() {
        let commands = parse_script("# warm the cache\n\nunpin_table lineitem\n").unwrap();
        assert_eq!(
            commands,
            vec![AdminCommand::UnpinTable {
                name: "lineitem".to_string()
            }]
        );
    }

    #[test]
    fn multiple_commands_in_order() {
        let commands =
            parse_script("pin_table path=a.parquet tier=gpu name=a\nunpin_table b").unwrap();
        assert_eq!(commands.len(), 2);
    }

    #[test]
    fn rejects_missing_tier() {
        let err = parse_script("pin_table path=a.parquet name=a").unwrap_err();
        assert!(err.contains("tier"), "{err}");
    }

    #[test]
    fn rejects_missing_path_for_parquet() {
        let err = parse_script("pin_table tier=gpu name=a").unwrap_err();
        assert!(err.contains("path"), "{err}");
    }

    #[test]
    fn rejects_bad_tier_unknown_verb_and_duplicate_key() {
        assert!(parse_script("pin_table path=a tier=vram name=a").is_err());
        assert!(parse_script("drop_all_tables now").is_err());
        assert!(parse_script("pin_table path=a path=b tier=gpu name=a").is_err());
    }

    #[test]
    fn rejects_empty_script_and_names_line_numbers() {
        assert!(parse_script("# nothing\n\n").is_err());
        let err = parse_script("unpin_table a\nbogus").unwrap_err();
        assert!(err.starts_with("line 2:"), "{err}");
    }
}
