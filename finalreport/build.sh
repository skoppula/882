#!/bin/bash

file="finalreport"
author1="karren"
author2="skoppula"
totalfile="$file-$author1-$author2"

# Compile the tex file
pdflatex --shell-escape -jobname=$totalfile paper.tex
bibtex $totalfile
pdflatex --shell-escape -jobname=$totalfile paper.tex
pdflatex --shell-escape -jobname=$totalfile paper.tex

if [ "$(uname)" == "Darwin" ]; then
    open ${totalfile}.pdf &
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    evince ${totalfile}.pdf &
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    echo 'Windows not supported!'
fi

rm *.aux *.log *.out *.bbl *.blg
