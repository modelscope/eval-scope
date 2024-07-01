# Copyright (c) Alibaba, Inc. and its affiliates.

from llmuses.third_party.toolbench_static import run_task


def main():
    # Construct the task config dictionary
    task_cfg_d: dict = dict(

    )

    # Run the task
    run_task(task_cfg=task_cfg_d)


if __name__ == '__main__':
    main()
