#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Narrow, anchor-counted additions to copies of the actual reactor source."""

def instrument(source):
    edits = []

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('source anchor not unique: ' + old[:100])
        source = source.replace(old, new)
        edits.append((old, new))
    replace('#include "io/rest/rest_reactor.hpp"', '#include "io/rest/rest_reactor.hpp"\n#include "d1_observer.hpp"')

    def obs(kind, req='request', tail=''):
        return f'd1::observe("{kind}", {req}, _ctx, _config, redirects(), !conditional_tag({req}.op->obj.get()).empty(){tail});'
    replace('          select_route(*slot.req);', '          ' + obs('select_begin', '(*slot.req)') + '\n          select_route(*slot.req);\n          ' + obs('select_end', '(*slot.req)'))
    replace('          slot.attempt_active = true;', '          slot.attempt_active = true;\n          ' + obs('submitted', '(*slot.req)'))
    anchor = '      auto const io_rng = op.io_rng;'
    replace(anchor, anchor + '\n      ' + obs('finish', tail=', http_status, curl_status, slot.sink.total_received'))
    old = '      if (request.route_cached && cached_route_refused(curl_status, http_status)) {'
    new = '      bool const d1_refused = request.route_cached && cached_route_refused(curl_status, http_status);\n      ' + obs('classification', tail=', http_status, curl_status, slot.sink.total_received, d1_refused') + '\n      if (d1_refused) {'
    replace(old, new)
    old = '            rest_obj->drop_route(request.route_snapshot);'
    replace(old, old + '\n            ' + obs('invalidated'))
    old = '          seed_route(*op.obj, request.delegated_url);'
    replace(old, '          ' + obs('seed_begin') + '\n          bool const d1_seeded = seed_route(*op.obj, request.delegated_url);\n          ' + obs('seed_end', tail=', 0, 0, 0, d1_seeded'))
    reverse = source
    for (old, new) in reversed(edits):
        reverse = reverse.replace(new, old)
    return (source, edits, reverse)
