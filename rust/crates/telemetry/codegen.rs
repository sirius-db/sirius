// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local schema adapters until Quent provides analysis code generation.
use heck::{ToSnakeCase, ToUpperCamelCase};
use proc_macro2::TokenStream;
use quent_fsm::Fsm;
use quent_resource::Resource;
use quent_schema::{DataType, Field, Record, Schema};
use quote::{format_ident, quote};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn usage_record<'a>(schema: &'a Schema, field: &Field) -> Result<Option<&'a Record>> {
    let DataType::EntityRef {
        data: Some(data), ..
    } = field.ty()
    else {
        return Ok(None);
    };
    let DataType::Record(path) = data.as_ref() else {
        return Ok(None);
    };
    let record = schema.record(path).ok_or("missing usage record")?;
    let Some(annotation) = record.annotations().constraint(Resource::NAME) else {
        return Ok(None);
    };
    Ok(matches!(
        serde_json::from_str::<Resource>(annotation.data().ok_or("missing resource annotation")?)?,
        Resource::Usage { .. }
    )
    .then_some(record))
}

fn attribute(ty: &DataType, name: &str, value: TokenStream) -> Result<TokenStream> {
    if let DataType::Option(inner) = ty {
        if matches!(**inner, DataType::Option(_)) {
            let inner = attribute(inner, name, quote!(value))?;
            return Ok(quote!(#value.as_ref().and_then(|value| #inner)));
        }
        let inner = attribute_value(inner, name, quote!(value))?;
        return Ok(quote!(#value.as_ref().map(|value| #inner)));
    }
    let expression = attribute_value(ty, name, value)?;
    Ok(quote!(Some(#expression)))
}

fn attribute_value(ty: &DataType, name: &str, value: TokenStream) -> Result<TokenStream> {
    let expression = match ty {
        DataType::String => quote!(DynamicAttribute::string(#name, #value)),
        DataType::EntityRef { .. } => {
            quote!(DynamicAttribute::string(#name, #value.target.to_string()))
        }
        DataType::Uuid => quote!(DynamicAttribute::string(#name, #value.to_string())),
        // Preserve the established viewer representation of bools and lists.
        DataType::Bool => quote!(DynamicAttribute::u8(#name, u8::from(*#value))),
        DataType::List(_) => quote!(DynamicAttribute::string(#name, format!("{:?}", #value))),
        DataType::U8
        | DataType::U16
        | DataType::U32
        | DataType::U64
        | DataType::I8
        | DataType::I16
        | DataType::I32
        | DataType::I64
        | DataType::F32
        | DataType::F64 => {
            let method = format_ident!("{}", format!("{ty:?}").to_lowercase());
            quote!(DynamicAttribute::#method(#name, *#value))
        }
        other => return Err(format!("unsupported FSM attribute {name}: {other:?}").into()),
    };
    Ok(expression)
}

/// Activity display aliases are application presentation policy.
/// Attributes follow their declaration order in the schema.
fn aliases(entity: &str, state: &str) -> Vec<String> {
    if entity == "QueryActivity" && state == "started" {
        vec!["working".into(), "waiting".into()]
    } else {
        vec![state.into()]
    }
}

pub fn generate(schema: &Schema, ui_entities: &[&str]) -> Result<(TokenStream, TokenStream)> {
    let mut stored = TokenStream::new();
    let mut ui = TokenStream::new();
    for entity in schema.entities() {
        let Some(fsm) = Fsm::try_from_entity(entity)? else {
            continue;
        };
        let name = entity.path().name().to_string();
        let type_name = name.to_snake_case();
        let event_type = format_ident!("{name}Event");
        let initial = format_ident!("{}", fsm.initial_state().to_upper_camel_case());
        let mut sequences = Vec::new();
        let mut finals = Vec::new();
        let mut next = Vec::new();
        let mut names = Vec::new();
        let mut usages = Vec::new();
        let mut attributes = Vec::new();
        let mut ui_states = Vec::new();
        for event in entity.events() {
            let state = event.name().to_string();
            let variant = format_ident!("{}", state.to_upper_camel_case());
            sequences.push(quote!(Self::#variant { seq, .. } => *seq));
            if fsm.is_final_state(event.name()) {
                finals.push(quote!(Self::#variant { .. }));
            }
            let targets: Vec<_> = fsm
                .transitions()
                .iter()
                .filter(|t| t.source() == event.name())
                .map(|t| format_ident!("{}", t.target().to_upper_camel_case()))
                .collect();
            let accepts = if targets.is_empty() {
                quote!(false)
            } else {
                quote!(matches!(next, #(Self::#targets { .. })|*))
            };
            next.push(quote!(Self::#variant { .. } => #accepts));
            names.push(quote!(Self::#variant { .. } => #state));
            let mut used_fields = Vec::new();
            let mut claims = Vec::new();
            let mut attr_fields = Vec::new();
            for field in event.fields() {
                let field_name = field.name().to_string();
                let ident = format_ident!("{field_name}");
                if let Some(record) = usage_record(schema, field)? {
                    let field_ident = &ident;
                    let capacities: Vec<_> = record.fields().map(|capacity| {
                        let name = capacity.name().to_string();
                        let ident = format_ident!("{name}");
                        match capacity.ty() {
                            DataType::U64 => Ok(quote!(CapacityValue::new(#name, #field_ident.data.#ident))),
                            DataType::Option(inner) if **inner == DataType::U64 => Ok(quote!(CapacityValue { name: #name, value: #field_ident.data.#ident })),
                            _ => Err("unsupported usage capacity"),
                        }
                    }).collect::<std::result::Result<_, _>>()?;
                    let capacities = if capacities.is_empty() {
                        quote!(smallvec![CapacityValue::new("unit", 1)])
                    } else {
                        quote!(smallvec![#(#capacities),*])
                    };
                    claims.push(quote!(AnalyzedUsage { resource_id: #ident.target, capacities: #capacities }));
                    used_fields.push(ident);
                } else if field_name != "seq"
                    && !(name == "QueryActivity" && field_name == "query_id")
                {
                    attr_fields.push(field);
                }
            }
            let attr_names: Vec<_> = attr_fields
                .iter()
                .map(|f| format_ident!("{}", f.name().to_string()))
                .collect();
            let attr_values: Vec<_> = attr_fields
                .iter()
                .map(|f| {
                    let ident = format_ident!("{}", f.name().to_string());
                    attribute(f.ty(), f.name(), quote!(#ident))
                })
                .collect::<Result<_>>()?;
            let attr_body = if attr_names.is_empty() {
                quote!(Vec::new())
            } else {
                quote!([#(#attr_values),*].into_iter().flatten().collect())
            };
            attributes.push(quote!(Self::#variant { #(#attr_names,)* .. } => #attr_body));
            usages.push(quote!(Self::#variant { #(#used_fields,)* .. } => smallvec![#(#claims),*]));
            if ui_entities.contains(&name.as_str()) {
                let state_names = aliases(&name, &state);
                let used_names: Vec<_> = used_fields.iter().map(ToString::to_string).collect();
                for alias in state_names {
                    ui_states.push(quote!(FsmStateTypeDecl { name: #alias.into(), usages: vec![#(#used_names.into()),*] }));
                }
            }
        }
        let display_name = if name == "QueryActivity" {
            quote!(crate::boilerplate::activity_state_name(self))
        } else {
            quote!(match self { #(#names,)* })
        };
        stored.extend(quote! {
            impl TransitionEvent for #event_type {
                fn sequence(&self) -> u16 { match self { #(#sequences,)* } }
                fn is_initial(&self) -> bool { matches!(self, Self::#initial { .. }) }
                fn is_final(&self) -> bool { matches!(self, #(#finals)|*) }
                fn is_valid_next(&self, next: &Self) -> bool { match self { #(#next,)* } }
                fn name(&self) -> &'static str { #display_name }
                fn usages(&self) -> SmallVec<[AnalyzedUsage; 1]> { match self { #(#usages,)* } }
            }
            impl #event_type {
                pub const TYPE_NAME: &'static str = #type_name;
                pub fn attributes(&self) -> Vec<DynamicAttribute> { match self { #(#attributes,)* } }
            }
        });
        if ui_entities.contains(&name.as_str()) {
            let declaration = format_ident!("{type_name}_declaration");
            let mut ui_transitions = Vec::new();
            for alias in aliases(&name, fsm.initial_state()) {
                ui_transitions.push(quote!(FsmTransitionDecl::Entry(#alias.into())));
            }
            for transition in fsm.transitions() {
                for source in aliases(&name, transition.source()) {
                    for target in aliases(&name, transition.target()) {
                        ui_transitions.push(
                            quote!(FsmTransitionDecl::Transition(#source.into(), #target.into())),
                        );
                    }
                }
            }
            for event in entity.events().filter(|e| fsm.is_final_state(e.name())) {
                for alias in aliases(&name, event.name()) {
                    ui_transitions.push(quote!(FsmTransitionDecl::Exit(#alias.into())));
                }
            }
            ui.extend(quote! {
                pub(super) fn #declaration() -> FsmTypeDecl {
                    FsmTypeDecl { name: #type_name.into(), states: vec![#(#ui_states),*],
                        transitions: vec![#(#ui_transitions),*] }
                }
            });
        }
    }
    Ok((stored, ui))
}

pub fn write(file: &str, tokens: TokenStream) -> Result<()> {
    let source = prettyplease::unparse(&syn::parse2(tokens)?);
    std::fs::write(
        std::path::Path::new(&std::env::var("OUT_DIR")?).join(file),
        source,
    )?;
    Ok(())
}
