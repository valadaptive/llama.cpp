#include "server-register.h"

#include "server-connect.h"
#include "server-context.h"
#include "server-mcp.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-tools.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

server_http_context::handler_t server_ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const common_json_error & e) {
            // JSON parse and type errors are invalid requests (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}


bool server_prepare_params(common_params & params, server_role_type role_type) {
    if (role_type != SERVER_ROLE_ROUTER) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // size the KV pool from --kv-unified-per-slot, unless the user pinned it with -c
    // or with -c 0 for max context
    const bool ctx_pool_auto_sized = params.kv_unified_per_slot > 0 &&
                                     params.n_ctx == 0 &&
                                     (uint32_t) params.fit_params_min_ctx != UINT32_MAX;

    if (ctx_pool_auto_sized) {
        params.n_ctx = params.n_parallel * params.kv_unified_per_slot;
        SRV_INF("--kv-unified-per-slot: sizing KV pool to n_parallel * kv_unified_per_slot = %d * %d = %d\n", params.n_parallel,
                params.kv_unified_per_slot, params.n_ctx);
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // check early, so a missing llama-connect fails-fast
    if (params.server_connect) {
        const std::string reason = server_connect::unavailable_reason(params);
        if (!reason.empty()) {
            SRV_ERR("--connect is not available: %s\n", reason.c_str());
            return false;
        }
        params.server_connect_code = server_connect::resolve_code(params.server_connect_code);
    } else if (!params.server_connect_code.empty()) {
        // the code is only meaningful while the tunnel runs, and /props keys off it being set
        SRV_WRN("%s", "--connect-code is ignored without --connect\n");
        params.server_connect_code.clear();
    }

    return true;
}


void server_register_routes(server_http_context & ctx_http, server_routes & routes) {
    ctx_http.get ("/health",                   server_ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                server_ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  server_ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    server_ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    server_ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   server_ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",                server_ex_wrapper(routes.get_models));
    ctx_http.post("/completion",               server_ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              server_ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           server_ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         server_ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      server_ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", server_ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             server_ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                server_ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  server_ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     server_ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/tts",                      server_ex_wrapper(routes.post_tts));
    ctx_http.post("/v1/audio/speech",          server_ex_wrapper(routes.post_speech_oai));
    ctx_http.post("/audio/speech",             server_ex_wrapper(routes.post_speech_oai));
    ctx_http.ws  ("/v1/realtime",              routes.ws_realtime);
    ctx_http.post("/v1/messages",              server_ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   server_ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                server_ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               server_ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            server_ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   server_ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                server_ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                server_ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             server_ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/systemone",             server_ex_wrapper(routes.post_systemone));
    ctx_http.post("/tokenize",                 server_ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               server_ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           server_ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    server_ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", server_ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           server_ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        server_ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         server_ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            server_ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            server_ex_wrapper(routes.post_lora_adapters));
    // Control vector hotswap
    ctx_http.get ("/cvectors",                 server_ex_wrapper(routes.get_cvectors));
    ctx_http.post("/cvectors",                 server_ex_wrapper(routes.post_cvectors));
    ctx_http.post("/cvectors/load",            server_ex_wrapper(routes.post_cvectors_load));
    ctx_http.post("/cvectors/remove",          server_ex_wrapper(routes.post_cvectors_remove));
    // Save & load slots
    ctx_http.get ("/slots",                    server_ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           server_ex_wrapper(routes.post_slots));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();
}


void server_register_stream_routes(
        server_http_context & ctx_http,
        const server_http_context::handler_t & stream_get,
        const server_http_context::handler_t & streams_lookup,
        const server_http_context::handler_t & stream_delete) {
    ctx_http.get ("/v1/stream",                server_ex_wrapper(stream_get));
    ctx_http.post("/v1/streams/lookup",        server_ex_wrapper(streams_lookup));
    ctx_http.del ("/v1/stream",                server_ex_wrapper(stream_delete));
}


bool server_setup_features(
        server_http_context & ctx_http,
        const common_params & params,
        server_mcp & mcp_mgr,
        server_tools & tools,
        server_role_type role_type) {
    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    if (params.cors_origins == "*" && params.api_keys.empty()) {
        SRV_WRN("%s", "security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/llama.cpp/pull/25655)\n");
    }

    std::vector<std::string> warn_names;
    if (role_type == SERVER_ROLE_ROUTER) {
        warn_names.push_back("router mode");
    }

    if (params.server_connect) {
        warn_names.push_back("peer-to-peer tunnel (llama-connect, experimental)");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.ui_mcp_proxy) {
        ctx_http.get ("/cors-proxy",      server_ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      server_ex_wrapper(proxy_handler_post));
        warn_names.push_back("MCP proxy (experimental)");
    } else {
        ctx_http.get ("/cors-proxy",      server_ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      server_ex_wrapper(res_403));
    }

    try {
        mcp_mgr.start(params);
    } catch (const std::exception & e) {
        SRV_ERR("MCP starting failed: %s\n", e.what());
        return false;
    }

    if (!params.server_tools.empty() || !mcp_mgr.empty()) {
        try {
            tools.setup(params.server_tools, mcp_mgr, params.server_tools_runtime);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return false;
        }
        ctx_http.get ("/tools",           server_ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           server_ex_wrapper(tools.handle_post));
        if (!params.server_tools.empty()) {
            warn_names.push_back("server tools (experimental)");
        }
        if (!params.server_tools_runtime.empty()) {
            warn_names.push_back("tools runtime (experimental)");
        }
        if (!mcp_mgr.empty()) {
            warn_names.push_back("MCP servers (experimental)");
        }
    } else {
        ctx_http.get ("/tools",           server_ex_wrapper(res_403));
        ctx_http.post("/tools",           server_ex_wrapper(res_403));
    }

    if (!warn_names.empty()) {
        std::string features;
        for (const auto & name : warn_names) {
            if (!features.empty()) features += ", ";
            features += name;
        }
        SRV_WRN("security: %s enabled - do not expose to untrusted environments\n", features.c_str());
    }

    return true;
}
