#!/usr/bin/env bash

SDIR=$1

# render shader
./../../bin/dd_shader_reflect -o svis_shader_enums.h -e RE_LineDot \
	-v "$SDIR/LineDot_V.vert" \
	-f "$SDIR/LineDot_F.frag"
