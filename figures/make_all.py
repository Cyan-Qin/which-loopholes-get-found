"""Rebuild every paper figure from the committed results: python -m figures.make_all

The overview (Figure 2) is TikZ and needs a TeX installation; the other figures need only matplotlib.
"""

from figures import gen_fig_auroc, gen_fig_fpr_heatmap, gen_fig_h1, gen_fig_h2_h3, gen_fig_overview, gen_fig_takeover

if __name__ == "__main__":
    for module in (gen_fig_takeover, gen_fig_overview, gen_fig_fpr_heatmap, gen_fig_h1, gen_fig_auroc, gen_fig_h2_h3):
        module.main()
