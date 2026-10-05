import os, argparse, pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch

def wrap_text(c, text, x, y, max_width, leading=12, font='Helvetica', size=10):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    c.setFont(font, size)
    words, line = text.split(), ''
    while words:
        w = words.pop(0)
        test = (line + ' ' + w).strip()
        if stringWidth(test, font, size) < max_width:
            line = test
        else:
            c.drawString(x, y, line); y -= leading; line = w
    if line: c.drawString(x, y, line); y -= leading
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--plots_dir', default=None)
    args = ap.parse_args()

    plots = args.plots_dir or os.path.join(args.results_dir, 'plots')
    fig1 = os.path.join(plots, 'fig_accuracy_vs_deleted.png')
    fig2 = os.path.join(plots, 'fig_time_saved_vs_deleted.png')
    fig3 = os.path.join(plots, 'fig_mia_auc.png')
    table_csv = os.path.join(plots, 'table_layout_times.csv')

    c = canvas.Canvas(args.out, pagesize=letter)
    width, height, margin = letter[0], letter[1], 54

    c.setFont('Helvetica-Bold', 16)
    c.drawString(margin, height - margin, 'SISA Machine Unlearning — Report')
    y = height - margin - 24
    y = wrap_text(c,
        "We implement SISA (Sharded, Isolated, Sliced, Aggregated) training to support "
        "selective unlearning without full retraining. We quantify accuracy, time saved, and "
        "membership-privacy risk (AUC) before/after deletions.",
        margin, y, width - 2*margin, leading=12)
    c.showPage()

    c.setFont('Helvetica-Bold', 14); c.drawString(margin, height - margin, 'Fig-1: Test accuracy vs % deleted')
    if os.path.isfile(fig1): c.drawImage(ImageReader(fig1), margin, height - 9*inch, width=6.5*inch, preserveAspectRatio=True, mask='auto')
    c.showPage()

    c.setFont('Helvetica-Bold', 14); c.drawString(margin, height - margin, 'Fig-2: Time saved vs % deleted')
    if os.path.isfile(fig2): c.drawImage(ImageReader(fig2), margin, height - 5.5*inch, width=6.5*inch, preserveAspectRatio=True, mask='auto')
    c.setFont('Helvetica-Bold', 14); c.drawString(margin, height - 6*inch, 'Fig-3: MIA AUC comparison')
    if os.path.isfile(fig3): c.drawImage(ImageReader(fig3), margin, height - 10.5*inch, width=3.25*inch, preserveAspectRatio=True, mask='auto')
    c.showPage()

    c.setFont('Helvetica-Bold', 14); c.drawString(margin, height - margin, 'Table-1: K×S layout, retrained slices, wall-clock time')
    y = height - margin - 18
    if os.path.isfile(table_csv):
        df = pd.read_csv(table_csv).head(28)
        text = df.to_string(index=False, max_cols=6, max_rows=28)
        y = wrap_text(c, text, margin, y, width - 2*margin, leading=11, font='Courier', size=8)
    y -= 12
    c.setFont('Helvetica-Bold', 12); c.drawString(margin, y, 'Foundation-model tie-in (Adapters/LoRA):'); y -= 14
    y = wrap_text(c,
        "SISA’s shard/slice isolation maps cleanly onto adapter-style fine-tunes: train one base model, "
        "keep per-shard adapters (or LoRA matrices) and re-train only the impacted adapters from the earliest "
        "affected slice while freezing everything else. At inference, ensemble adapters via logit averaging or "
        "gating. This preserves unlearning efficiency (localized updates) and minimizes accuracy loss—consistent with "
        "LoRA’s aim of reducing trainable parameters and GPU memory.", margin, y, width - 2*margin)
    c.save()

if __name__ == '__main__':
    main()
