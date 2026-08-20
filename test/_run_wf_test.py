import sys, io, os

BASE = r"d:\AI大模型应用开发实战\8-Agent的自主规划与工具开发\CASE-投顾AI助手（混合式）"
os.chdir(BASE)
sys.path.insert(0, BASE)

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

old_stdout = sys.stdout
sys.stdout = io.StringIO()

import unittest
suite = unittest.TestLoader().discover('test', pattern='test_workflow*')
buf = io.StringIO()
r = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)

sys.stdout = old_stdout

result_path = os.path.join(BASE, '_wf_result.txt')
with open(result_path, 'w', encoding='utf-8') as f:
    f.write(buf.getvalue())
    f.write(f"\n\nSUMMARY: R={r.testsRun} F={len(r.failures)} E={len(r.errors)} S={len(r.skipped)}\n")
    for t, tb in r.failures:
        f.write(f"\nFAIL: {t}\n{tb}\n")
    for t, tb in r.errors:
        f.write(f"\nERROR: {t}\n{tb}\n")

print(f"Done. R={r.testsRun} F={len(r.failures)} E={len(r.errors)} S={len(r.skipped)}")
