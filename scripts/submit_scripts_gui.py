#!/usr/bin/env python3
"""Minimal GUI for editing and submitting the main Slurm job scripts."""

from __future__ import annotations

import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent

SCRIPT_PATHS = {
    "New Data": ROOT / "submit_scripts" / "Submit_new_data.sh",
    "Multitarget RF": ROOT / "submit_scripts" / "Submit_multitarget_training.sh",
    "Single RF": ROOT / "submit_scripts" / "Submit_single_rf.sh",
    "Predict RF": ROOT / "submit_scripts" / "Submit_predict_rf.sh",
}


class ScriptTab:
    def __init__(self, parent: ttk.Notebook, label: str, path: Path, output_callback):
        self.label = label
        self.path = path
        self.output_callback = output_callback
        self.dirty = False

        self.frame = ttk.Frame(parent, padding=10)
        parent.add(self.frame, text=label)

        header = ttk.Frame(self.frame)
        header.pack(fill="x", pady=(0, 8))

        ttk.Label(header, text=str(path), font=("TkDefaultFont", 9)).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(header, text="Reload", command=self.reload).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Save As", command=self.save_as).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Save", command=self.save).pack(side="right")

        self.editor = tk.Text(
            self.frame,
            wrap="none",
            undo=True,
            font=("Courier", 11),
            padx=8,
            pady=8,
        )
        self.editor.pack(fill="both", expand=True)
        self.editor.bind("<<Modified>>", self._on_modified)

        xscroll = ttk.Scrollbar(self.frame, orient="horizontal", command=self.editor.xview)
        yscroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.editor.yview)
        self.editor.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        xscroll.pack(fill="x")
        yscroll.place(relx=1.0, rely=0.0, relheight=1.0, anchor="ne")

        self.reload()

    def _on_modified(self, _event=None):
        if self.editor.edit_modified():
            self.dirty = True
            self.output_callback(f"[edited] {self.path.name}")
            self.editor.edit_modified(False)

    def reload(self):
        try:
            content = self.path.read_text()
        except FileNotFoundError:
            content = ""
            self.output_callback(f"[missing] {self.path}")
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", content)
        self.editor.edit_modified(False)
        self.dirty = False
        self.output_callback(f"[loaded] {self.path}")

    def save(self):
        content = self.editor.get("1.0", "end-1c")
        self.path.write_text(content)
        self.dirty = False
        self.output_callback(f"[saved] {self.path}")

    def save_as(self):
        target = filedialog.asksaveasfilename(
            initialdir=str(self.path.parent),
            initialfile=self.path.name,
            title=f"Save copy of {self.path.name}",
        )
        if not target:
            return
        Path(target).write_text(self.editor.get("1.0", "end-1c"))
        self.output_callback(f"[saved copy] {target}")


class SubmitScriptsGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TL Project Submit Scripts")
        self.root.geometry("1150x760")

        self.tabs: list[ScriptTab] = []
        self.output: tk.Text | None = None

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))

        ttk.Button(controls, text="Save Current", command=self.save_current).pack(side="left")
        ttk.Button(controls, text="Submit Current", command=self.submit_current).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(controls, text="Run With Bash", command=self.run_current_bash).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(controls, text="Open Slurm Folder", command=self.show_slurm_dir).pack(
            side="left", padx=(8, 0)
        )

        hint = "Edits are made directly to the script files. Save before submitting."
        ttk.Label(controls, text=hint).pack(side="right")

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)

        for label, path in SCRIPT_PATHS.items():
            self.tabs.append(ScriptTab(self.notebook, label, path, self.log))

        output_frame = ttk.LabelFrame(outer, text="Command Output", padding=8)
        output_frame.pack(fill="both", expand=False, pady=(10, 0))

        self.output = tk.Text(
            output_frame,
            height=12,
            wrap="word",
            state="disabled",
            font=("Courier", 10),
            padx=8,
            pady=8,
        )
        self.output.pack(fill="both", expand=True)

        self.root.bind("<Control-s>", lambda _event: self.save_current())

    def current_tab(self) -> ScriptTab:
        idx = self.notebook.index(self.notebook.select())
        return self.tabs[idx]

    def log(self, message: str):
        if self.output is None:
            return
        self.output.configure(state="normal")
        self.output.insert("end", f"{message}\n")
        self.output.see("end")
        self.output.configure(state="disabled")

    def save_current(self):
        self.current_tab().save()

    def _confirm_unsaved(self, tab: ScriptTab) -> bool | None:
        if not tab.dirty:
            return True
        return messagebox.askyesnocancel(
            "Unsaved changes",
            (
                f"{tab.path.name} has unsaved edits.\n\n"
                "Yes: save and run\n"
                "No: run the last saved version\n"
                "Cancel: stop"
            ),
        )

    def _run_command(self, command: list[str], cwd: Path):
        self.log(f"$ {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - GUI error path
            self.log(f"[error] {exc}")
            messagebox.showerror("Command failed", str(exc))
            return

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if stdout:
            self.log(stdout)
        if stderr:
            self.log(stderr)
        self.log(f"[exit code] {result.returncode}")

    def submit_current(self):
        tab = self.current_tab()
        decision = self._confirm_unsaved(tab)
        if decision is None:
            self.log("[cancelled] submit")
            return
        if decision:
            tab.save()
        self._run_command(["sbatch", tab.path.name], cwd=tab.path.parent)

    def run_current_bash(self):
        tab = self.current_tab()
        decision = self._confirm_unsaved(tab)
        if decision is None:
            self.log("[cancelled] bash run")
            return
        if decision:
            tab.save()
        warning = (
            "This runs the submit script directly with bash in your current environment.\n"
            "Use this only if you intentionally want to bypass sbatch."
        )
        if not messagebox.askyesno("Run with bash", warning):
            return
        self._run_command(["bash", tab.path.name], cwd=tab.path.parent)

    def show_slurm_dir(self):
        slurm_dir = ROOT / "submit_scripts" / "slurm_files"
        self.log(str(slurm_dir))
        messagebox.showinfo("Slurm Output", str(slurm_dir))


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    SubmitScriptsGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
