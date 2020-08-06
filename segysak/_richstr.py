"""Fucntions to help with pretty printing in IPython
"""


def _prettify(self, p, cycle):
    if cycle:
        return p.text(self)
    else:
        lines = self.splitlines()
        p.text("Text Header")
        for line in lines:
            p.text(line)
            p.break_()


def _htmlify(self):
    html = f"<h3 style='font-size: medium;'>Text Header</h3>"
    lines = self.split("\n")
    html += "<h3 style='font-size: small;'>"
    for line in lines:
        html += f"{line}<br/>"
    html += "</h3>"
    return html


def _upgrade_txt_richstr(text):
    return type(
        "rich_texthead", (str,), dict(_repr_pretty_=_prettify, _repr_html_=_htmlify)
    )(text)

