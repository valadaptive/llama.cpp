#pragma once

#include "server-common.h"
#include "server-http.h"

struct server_mcp;
struct server_routes;
struct server_tools;

// Convert handler exceptions into structured HTTP error responses.
server_http_context::handler_t server_ex_wrapper(server_http_context::handler_t func);

// Apply llama-server runtime policy after generic common parameter processing.
// Returns false when the selected role has an invalid configuration.
bool server_prepare_params(common_params & params, server_role_type role_type);

// Register the fixed API table shared by router, standalone, and embedded servers.
void server_register_routes(server_http_context & ctx_http, server_routes & routes);

// Register resumable-stream handlers. Routers pass proxy handlers; instances pass
// the local session-manager handlers.
void server_register_stream_routes(
        server_http_context & ctx_http,
        const server_http_context::handler_t & stream_get,
        const server_http_context::handler_t & streams_lookup,
        const server_http_context::handler_t & stream_delete);

// Start optional MCP/tool features and register their routes.
bool server_setup_features(
        server_http_context & ctx_http,
        const common_params & params,
        server_mcp & mcp_mgr,
        server_tools & tools,
        server_role_type role_type);
