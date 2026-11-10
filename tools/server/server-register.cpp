#include "server-register.h"

#include "server-common.h"
#include "server-context.h"
#include "server-stream.h"
#include "server-tools.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

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

void server_register_routes(
        server_http_context & ctx_http,
        server_routes       & routes,
        server_tools        * tools,
        server_http_context::handler_t stream_get,
        server_http_context::handler_t streams_lookup,
        server_http_context::handler_t stream_delete,
        server_http_context::handler_t cors_proxy_get,
        server_http_context::handler_t cors_proxy_post) {
    auto ex_wrapper = server_ex_wrapper;

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Control vectors hotswap
    ctx_http.get ("/cvectors",                 ex_wrapper(routes.get_cvectors));
    ctx_http.post("/cvectors",                 ex_wrapper(routes.post_cvectors));
    ctx_http.post("/cvectors/load",            ex_wrapper(routes.post_cvectors_load));
    ctx_http.post("/cvectors/remove",          ex_wrapper(routes.post_cvectors_remove));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // resumable streaming: a child binds the local session factories, the router binds
    // proxies that resolve the owning child
    if (!stream_get)     { stream_get     = server_stream_make_get_handler(); }
    if (!streams_lookup) { streams_lookup = server_stream_make_lookup_handler(); }
    if (!stream_delete)  { stream_delete  = server_stream_make_delete_handler(); }
    ctx_http.get ("/v1/stream",                ex_wrapper(stream_get));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup));
    ctx_http.del ("/v1/stream",                ex_wrapper(stream_delete));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

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

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    ctx_http.get ("/cors-proxy",      ex_wrapper(cors_proxy_get  ? cors_proxy_get  : res_403));
    ctx_http.post("/cors-proxy",      ex_wrapper(cors_proxy_post ? cors_proxy_post : res_403));

    // EXPERIMENTAL built-in tools
    if (tools) {
        ctx_http.get ("/tools",           ex_wrapper(tools->handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools->handle_post));
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }
}
