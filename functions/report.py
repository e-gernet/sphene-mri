"""
Small tool to build a self-contained HTML report: text, figures, tables and checks.

The report is one ``.html`` file with the figures embedded (so it can be sent or
opened anywhere), next to a ``figures/`` folder with the same images and a
``controles.csv`` listing every check. Used by ``validation.py``.

Usage
-----
>>> rep = Report("validation_report", "Mon rapport")
>>> sec = rep.section("Titre", "La question posée en une phrase")
>>> sec.text("Un paragraphe.")
>>> sec.figure(fig, "figure1.png", "Légende")
>>> sec.table(["a", "b"], [[1, 2], [3, 4]])
>>> sec.check("Biais", "0,3 %", "moins de 1 %", True)
>>> rep.render()
"""

import base64
import csv
import html
from datetime import datetime
from pathlib import Path

PASS, FAIL, WARN, INFO = "RÉUSSI", "ÉCHEC", "ATTENTION", "INFO"

_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1f2933;
       max-width: 980px; margin: 24px auto; padding: 0 16px; line-height: 1.5; }
h1 { font-size: 26px; margin-bottom: 4px; } h2 { font-size: 21px; margin-top: 38px; border-bottom: 2px solid #dde3ea;
     padding-bottom: 4px; } h3 { font-size: 16px; margin-bottom: 4px; }
.sub { color: #52606d; margin-top: 0; }
.question { background: #eef4fb; border-left: 4px solid #4C72B0; padding: 8px 12px; margin: 10px 0; }
.conclusion { background: #f1f8f1; border-left: 4px solid #55A868; padding: 8px 12px; margin: 14px 0; }
.box { background: #fafbfc; border: 1px solid #dde3ea; border-radius: 6px; padding: 8px 14px; margin: 12px 0; }
table { border-collapse: collapse; margin: 10px 0; font-size: 13px; width: 100%; }
th, td { border: 1px solid #dde3ea; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f0f3f7; } tr:nth-child(even) td { background: #fafbfc; }
figure { margin: 14px 0; } figure img { max-width: 100%; border: 1px solid #e5e9ee; }
figcaption { font-size: 13px; color: #52606d; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; color: white; font-size: 12px;
         font-weight: 600; white-space: nowrap; }
.RÉUSSI { background: #2e8b57; } .ÉCHEC { background: #c0392b; } .ATTENTION { background: #d98e04; }
.INFO { background: #7b8794; }
.small { font-size: 13px; color: #52606d; }
"""


class Section:
    """One section of the report (created with :meth:`Report.section`)."""

    def __init__(self, report, title, question):
        self.report, self.title, self.question = report, title, question
        self.blocks = []          # ("text", html) | ("figure", name, png bytes, caption) | ("table", header, rows)
        self.checks = []          # dicts: name, value, rule, status
        self.conclusion_text = ""

    def text(self, content):
        """Add a paragraph (simple inline HTML such as ``<b>`` is allowed)."""
        self.blocks.append(("text", content))

    def figure(self, fig, name, caption=""):
        """Save a matplotlib figure into ``figures/<name>`` and add it to the section."""
        import io

        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        png = buf.getvalue()
        (self.report.fig_dir / name).write_bytes(png)
        self.blocks.append(("figure", name, png, caption))

    def table(self, header, rows):
        """Add a table (``header``: list of str, ``rows``: list of lists)."""
        self.blocks.append(("table", header, rows))

    def check(self, name, value, rule, ok, level=None):
        """Record a pass/fail check. ``level="warn"`` makes a failure a warning only."""
        status = PASS if ok else (WARN if level == "warn" else FAIL)
        self.checks.append({"name": name, "value": value, "rule": rule, "status": status})
        return ok

    def info(self, name, value):
        """Record a figure of interest that is neither passed nor failed."""
        self.checks.append({"name": name, "value": value, "rule": "", "status": INFO})

    def conclusion(self, content):
        """Set the take-home message of the section."""
        self.conclusion_text = content


class Report:
    """A report made of numbered :class:`Section` objects."""

    def __init__(self, out_dir, title, subtitle=""):
        self.out_dir = Path(out_dir)
        self.fig_dir = self.out_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.title, self.subtitle = title, subtitle
        self.sections = []
        self.intro = []           # html blocks shown before the summary
        self.summary_lines = []   # one-line take-home messages for the top of the report

    def section(self, title, question):
        """Start a new section."""
        sec = Section(self, title, question)
        self.sections.append(sec)
        return sec

    def all_checks(self):
        """List of ``(section number, section title, check)``."""
        return [(i + 1, s.title, c) for i, s in enumerate(self.sections) for c in s.checks]

    def counts(self):
        """Number of checks per status."""
        out = {PASS: 0, FAIL: 0, WARN: 0, INFO: 0}
        for _, _, c in self.all_checks():
            out[c["status"]] += 1
        return out

    # ── Output ────────────────────────────────────────────────────────────────

    def write_csv(self):
        path = self.out_dir / "controles.csv"
        # utf-8-sig: the byte-order mark lets Excel show the accents correctly (pandas ignores it)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["section", "titre", "controle", "valeur", "regle", "resultat"])
            for num, title, c in self.all_checks():
                w.writerow([num, title, c["name"], c["value"], c["rule"], c["status"]])
        return path

    @staticmethod
    def _badge(status):
        return f'<span class="badge {status}">{status}</span>'

    def _render_section(self, num, sec):
        out = [f'<h2 id="s{num}">{num}. {sec.title}</h2>',
               f'<div class="question"><b>La question.</b> {sec.question}</div>']
        for block in sec.blocks:
            if block[0] == "text":
                out.append(f"<p>{block[1]}</p>")
            elif block[0] == "figure":
                _, name, png, caption = block
                b64 = base64.b64encode(png).decode("ascii")
                out.append(f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(name)}">'
                           f"<figcaption>{caption}</figcaption></figure>")
            else:
                _, header, rows = block
                head = "".join(f"<th>{h}</th>" for h in header)
                body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
                out.append(f"<table><tr>{head}</tr>{body}</table>")
        if sec.checks:
            rows = "".join(
                f"<tr><td>{self._badge(c['status'])}</td><td>{c['name']}</td><td>{c['value']}</td>"
                f"<td>{c['rule']}</td></tr>" for c in sec.checks)
            out.append("<h3>Contrôles</h3><table><tr><th></th><th>Contrôle</th><th>Valeur mesurée</th>"
                       f"<th>Critère</th></tr>{rows}</table>")
        if sec.conclusion_text:
            out.append(f'<div class="conclusion"><b>À retenir.</b> {sec.conclusion_text}</div>')
        return "\n".join(out)

    def render(self):
        """Write ``rapport.html`` and ``controles.csv``; return the path of the html file."""
        counts = self.counts()
        verdict = (f"{counts[PASS]} contrôles réussis, {counts[FAIL]} échec(s), "
                   f"{counts[WARN]} attention(s), {counts[INFO]} information(s)")
        parts = [f"<h1>{self.title}</h1>",
                 f'<p class="sub">{self.subtitle} — généré le {datetime.now():%d/%m/%Y à %H:%M}</p>']
        toc = " · ".join(f'<a href="#s{i + 1}">{i + 1}. {s.title}</a>' for i, s in enumerate(self.sections))
        parts.append(f'<p class="small"><b>Sommaire :</b> <a href="#bilan">Bilan</a> · {toc}</p>')
        parts += [f'<div class="box">{b}</div>' for b in self.intro]
        parts.append('<h2 id="bilan">Bilan</h2>')
        parts.append(f"<p><b>{verdict}.</b></p>")
        if self.summary_lines:
            parts.append("<ul>" + "".join(f"<li>{s}</li>" for s in self.summary_lines) + "</ul>")
        rows = "".join(
            f"<tr><td>{num}</td><td>{title}</td><td>{c['name']}</td><td>{c['value']}</td>"
            f"<td>{self._badge(c['status'])}</td></tr>" for num, title, c in self.all_checks()
            if c["status"] != INFO)
        parts.append("<table><tr><th>§</th><th>Section</th><th>Contrôle</th><th>Valeur</th><th></th></tr>"
                     f"{rows}</table>")
        parts += [self._render_section(i + 1, s) for i, s in enumerate(self.sections)]
        doc = (f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>{self.title}</title>'
               f"<style>{_CSS}</style></head><body>" + "\n".join(parts) + "</body></html>")
        path = self.out_dir / "rapport.html"
        path.write_text(doc, encoding="utf-8")
        self.write_csv()
        return path
