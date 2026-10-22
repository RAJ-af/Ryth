import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_sft_data.py"

def sample(q="What is a list?", a="A list stores ordered items.", lang="en", provider="manual", batch="b1"):
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}],
            "provider":provider,"lang":lang,"topic":"python","angle":"intro","batch":batch}

class ValidatorCliTests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([sys.executable,str(SCRIPT),*map(str,args)],capture_output=True,text=True)

    def test_check_passes_valid_record(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td); (raw/"one.jsonl").write_text(json.dumps(sample())+"\n",encoding="utf-8")
            result=self.run_cli("check",raw)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_check_rejects_bad_role_and_model_mention(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td)
            obj=sample(a="ChatGPT can explain this.")
            (raw/"one.jsonl").write_text(json.dumps(obj)+"\n",encoding="utf-8")
            result=self.run_cli("check",raw)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("prohibited",result.stderr)
            obj=sample()
            obj["messages"][0]["role"]="system"
            (raw/"one.jsonl").write_text(json.dumps(obj)+"\n",encoding="utf-8")
            result=self.run_cli("check",raw)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("role='user'",result.stderr)

    def test_merge_outputs_messages_only_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td)/"raw"; out=Path(td)/"merged"; raw.mkdir()
            (raw/"en.jsonl").write_text(json.dumps(sample())+"\n",encoding="utf-8")
            result=self.run_cli("merge",raw,out)
            self.assertEqual(result.returncode,0,result.stderr)
            row=json.loads((out/"en.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(set(row),{"messages"})
            manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["by_language"]["en"],1)

    def test_merge_refuses_invalid_data(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td)/"raw"; out=Path(td)/"merged"; raw.mkdir()
            (raw/"bad.jsonl").write_text("{broken}\n",encoding="utf-8")
            result=self.run_cli("merge",raw,out)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((out/"en.jsonl").exists())

    def _check_rows(self,rows):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td); (raw/"a.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n",encoding="utf-8")
            return self.run_cli("check",raw)

    def test_integer_batch_is_accepted(self):
        r=self._check_rows([sample(batch=0)])
        self.assertEqual(r.returncode,0,r.stderr)

    def test_same_answer_different_question_rejected(self):
        r=self._check_rows([sample(q="What is a list?"),sample(q="Why use loops in code?")])
        self.assertNotEqual(r.returncode,0)
        self.assertIn("exact duplicate answer",r.stderr)

    def test_short_same_question_rejected(self):
        r=self._check_rows([sample(q="Hi",a="Hello there friend."),sample(q="Hi",a="Greetings to you today.")])
        self.assertNotEqual(r.returncode,0)
        self.assertIn("exact duplicate question",r.stderr)

    def test_hindi_needs_devanagari_in_both(self):
        r=self._check_rows([sample(q="list kya hai?",a="सूची एक डेटा संरचना है।",lang="hi")])
        self.assertNotEqual(r.returncode,0)
        self.assertIn("Hindi question must contain Devanagari",r.stderr)

if __name__=="__main__":
    unittest.main()
