#!/usr/bin/env bash
# Repack the edited _pptx_work/ folder back into main_pt_slides.pptx
set -e
cd "$(dirname "$0")"
cd _pptx_work
rm -f ../main_pt_slides.pptx
# [Content_Types].xml must be stored/first; zip everything.
zip -q -X -r ../main_pt_slides.pptx '[Content_Types].xml' _rels docProps ppt
echo "repacked -> main_pt_slides.pptx"
