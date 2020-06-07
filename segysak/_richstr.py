
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
    html = f"<h3>Text Header<h3/>"
    lines = self.split("\n")
    for line in lines:
        html += f"{line}<br/>"
    return html


def _upgrade_txt_richstr(text):
    return type("richstr", (str,), dict(_repr_pretty_=_prettify, _repr_html_=_htmlify))(
        text
    )