fig_width_pt = 384.0
inches_per_pt = 1.0/72.27
linewidth = fig_width_pt*inches_per_pt

def pgf_with_latex(nplots, hscale = 0.45):
    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "DejaVu Sans",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 5,               # LaTeX default is 10pt font.
        "font.size": 5,
        "legend.fontsize": 5,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "axes.titlesize": 5,
        # "figure.figsize": figsize(1.0, hscale, nplots),     # default fig size of 0.9 textwidth
        "pgf.preamble": "\n".join([
            # r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            # r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{relsize}",  # allows relative font sizing
            r"\usepackage{sfmath}",   # optional: makes math use sans-serif to match font.family
            r"\renewcommand{\rmdefault}{\sfdefault}",  # match math and text font
            r"\everymath{\scriptstyle}",  # make all math smaller by default       
            ])
        }
    return pgf_with_latex 