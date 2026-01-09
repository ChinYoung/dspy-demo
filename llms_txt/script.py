from pathlib import Path
from custom_lm.lms import Lm_Glm
from init_dspy import init_dspy

from llms_txt.RepoAnalyzer import RepositoryAnalyzer
from llms_txt.utils import gather_repository_info


def main():
    init_dspy(Lm_Glm)

    # Initialize our analyzer
    analyzer = RepositoryAnalyzer()

    # Gather DSPy repository information
    repo_url = "https://github.com/ChinYoung/gracias"
    file_tree, readme_content, package_files = gather_repository_info(repo_url)

    # Generate llms.txt
    result = analyzer(
        repo_url=repo_url,
        file_tree=file_tree,
        readme_content=readme_content,
        package_files=package_files,
    )

    # Save result to local file
    out_path = Path(__file__).resolve().parent / "llms.txt"
    out_path.write_text(str(result), encoding="utf-8")
    print(f"Wrote llms.txt to {out_path}")
    return result


if __name__ == "__main__":
    main()
