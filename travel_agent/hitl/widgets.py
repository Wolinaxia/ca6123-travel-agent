"""Clickable approval card for Jupyter / Colab (ipywidgets).

    from travel_agent.hitl.widgets import approval_widget
    ui = approval_widget(state, proposal, prefs)
    display(ui.box)
    # after clicking "Change", ui.change_message holds the text for the agent

Buttons: Approve (with a "I have read the warnings" tick box when needed),
Reject (with a reason), Change (ask for something else) and Why?. A rejection
may produce "Should I remember...?" suggestions with Yes / No buttons: memory
is only written when the user clicks Yes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from travel_agent.memory.preferences import PreferenceStore

from . import approval as hitl


@dataclass
class ApprovalUI:
    box: Any
    status: dict[str, Any] = field(default_factory=dict)
    change_message: str | None = None


def approval_widget(state: Any, proposal: hitl.Proposal, prefs: PreferenceStore | None = None) -> ApprovalUI:
    import ipywidgets as w
    from IPython.display import HTML, display

    card = w.HTML(hitl.render_card_html(proposal))
    ack = w.Checkbox(value=False, description="I have read the warnings", indent=False)
    ack.layout.display = "flex" if proposal.needs_human else "none"
    reason = w.Text(placeholder="Why? / what would you like instead?", layout=w.Layout(width="420px"))
    approve_btn = w.Button(description="Approve", button_style="success", icon="check")
    reject_btn = w.Button(description="Reject", button_style="danger", icon="times")
    change_btn = w.Button(description="Change", button_style="warning", icon="exchange")
    why_btn = w.Button(description="Why?", icon="question")
    out = w.Output()
    ui = ApprovalUI(box=None)

    def show_suggestions(suggestions: list[dict[str, Any]]) -> None:
        for suggestion in suggestions:
            yes = w.Button(description="Yes, remember", button_style="info")
            no = w.Button(description="No")
            label = w.HTML(f"<b>{suggestion['question']}</b>")

            def on_yes(_btn, s=suggestion):
                if prefs is not None:
                    prefs.confirm(s["id"])
                with out:
                    print(f"Saved. You can say 'forget {s['key']}' at any time.")

            def on_no(_btn, s=suggestion):
                if prefs is not None:
                    prefs.dismiss(s["id"])
                with out:
                    print("OK, not saved.")

            yes.on_click(on_yes)
            no.on_click(on_no)
            with out:
                display(w.VBox([label, w.HBox([yes, no])]))

    def on_approve(_btn):
        out.clear_output()
        with out:
            decision = hitl.approve(state, proposal, acknowledge_warnings=ack.value)
            if not decision["ok"]:
                print("Not approved:", decision["error"])
                return
            result = hitl.execute_approved(state, proposal)
            ui.status = result
            if result["ok"]:
                display(HTML(f"<b style='color:#1b7f3b'>{result['booking']['observation']}</b>"))
            else:
                print("Booking blocked:", result["error"])

    def on_reject(_btn):
        out.clear_output()
        result = hitl.reject(state, proposal, reason.value, prefs)
        ui.status = result
        with out:
            print("Rejected. Nothing was booked.")
        show_suggestions(result["suggestions"])

    def on_change(_btn):
        out.clear_output()
        result = hitl.request_change(state, proposal, reason.value or "show me other options", prefs)
        ui.status = result
        ui.change_message = result["message_for_agent"]
        if not result["ok"]:
            with out:
                print("Not sent:", result["error"])
            return
        with out:
            print("Change requested. Run the next cell to send this to the agent:")
            print(" ", result["message_for_agent"])
        show_suggestions(result["suggestions"])

    def on_why(_btn):
        out.clear_output()
        with out:
            print(hitl.explain(state, proposal))

    approve_btn.on_click(on_approve)
    reject_btn.on_click(on_reject)
    change_btn.on_click(on_change)
    why_btn.on_click(on_why)
    ui.box = w.VBox([card, ack, w.HBox([approve_btn, reject_btn, change_btn, why_btn]), reason, out])
    return ui
