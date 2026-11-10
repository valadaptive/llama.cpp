#pragma once

#include "server-http.h"

struct server_routes;
struct server_tools;

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
server_http_context::handler_t server_ex_wrapper(server_http_context::handler_t func);

// register every endpoint that a model-serving server exposes, whether it runs as the
// standalone llama-server binary or embedded in another application: the fixed API table
// bound to `routes`, the resumable-stream routes, /tools, /cors-proxy and the GCP compat
// aliases. null handler arguments fall back to the local implementations: the
// server_stream_make_*_handler() factories for the stream routes and a 403
// feature-disabled response for the cors-proxy routes (the router server passes proxies
// instead). `tools` is null unless built-in tools are enabled (403 response as well).
void server_register_routes(
        server_http_context & ctx_http,
        server_routes       & routes,
        server_tools        * tools           = nullptr,
        server_http_context::handler_t stream_get      = nullptr,
        server_http_context::handler_t streams_lookup  = nullptr,
        server_http_context::handler_t stream_delete   = nullptr,
        server_http_context::handler_t cors_proxy_get  = nullptr,
        server_http_context::handler_t cors_proxy_post = nullptr);
