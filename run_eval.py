"""
评测命令行入口
用法:
  python run_eval.py --mode retrieval        # 仅离线检索评测
  python run_eval.py --mode generation       # 仅生成评测(需配置 LLM_API_KEY)
  python run_eval.py --mode all              # 两者都跑(默认)
  python run_eval.py --mode retrieval --n 300 --ks 1 3 5 10
  python run_eval.py --mode generation --gen-n 30
  python run_eval.py --rebuild-set           # 强制重建评测集
报告输出: data/eval_report.json
"""
import argparse
import json
import sys

sys.path.insert(0, ".")

from src import eval as eval_mod


def main():
    p = argparse.ArgumentParser(description="bjhc RAG 评测")
    p.add_argument("--mode", choices=["retrieval", "generation", "all"],
                   default="all")
    p.add_argument("--n", type=int, default=200, help="评测集抽样数(检索)")
    p.add_argument("--gen-n", type=int, default=30, help="生成评测样本数")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10],
                   help="检索指标截断 k 列表")
    p.add_argument("--rebuild-set", action="store_true",
                   help="强制重建评测集")
    args = p.parse_args()

    print(f"开始评测 (mode={args.mode}, n={args.n}, ks={args.ks}) ...")
    report = eval_mod.run_eval(
        mode=args.mode, n=args.n, gen_n=args.gen_n,
        ks=args.ks, force_set=args.rebuild_set)

    if "retrieval" in report:
        r = report["retrieval"]
        print("\n===== 检索评测 (样本 %d) =====" % r.get("samples", 0))
        print("文件级 :", json.dumps(r.get("file_level", {}), ensure_ascii=False))
        print("片段级 :", json.dumps(r.get("chunk_level", {}), ensure_ascii=False))
    if "generation" in report:
        g = report["generation"]
        if g.get("skipped"):
            print("\n===== 生成评测 =====")
            print("已跳过:", g.get("reason"))
        else:
            print("\n===== 生成评测 (样本 %d) =====" % g.get("samples", 0))
            print("指标 :", json.dumps(g.get("metrics", {}), ensure_ascii=False))
    print("\n报告已保存:", eval_mod.REPORT_PATH)


if __name__ == "__main__":
    main()
