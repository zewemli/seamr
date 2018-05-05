#!/bin/bash
mkdir -p layouts
for X in ../layouts/*.svg; do
  echo $X
  node setDefs.js ../ontology/shape.svg $X > layouts/`basename $X`
done
mv ./layouts/*.svg ../layouts/
rm -Rf layouts
