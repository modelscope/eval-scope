# Copyright (c) Alibaba, Inc. and its affiliates.

import uuid
import time
from typing import Union

from evals.utils.logger import get_logger

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_openitag20220616 import models as open_itag_models
from evals.tools.itag.sdk.alpha_data_sdk.alpha_data_sdk import AlphaDataSdk
from evals.tools.itag.sdk.alpha_data_sdk import models as alphad_model
from evals.tools.itag.sdk.openitag_sdk.itag_sdk import ItagSdk

logger = get_logger(__name__)


class ItagManager(object):
    """
    iTag manager.

    Args:
        tenant_id: Tenant id.
        token: Token.
        employee_id: Employee id.

    Examples:
        >>> from evals.tools import ItagManager
        >>> itag_manager = ItagManager(tenant_id, token, employee_id)
        >>> itag_manager.create_dataset(dataset_file_path)
        >>> itag_manager.create_task(task_name, task_type, dataset_id, task_params)
        >>> itag_manager.get_task_result(task_id)
    """

    def __init__(self, tenant_id, token, employee_id, **kwargs):
        self._tenant_id = tenant_id
        self._token = token
        self._employee_id = employee_id

        self._itag = None
        self._alphad = None
        self._init_itag_client(tenant_id, token, employee_id)

    def _init_itag_client(self, tenant_id, token, employee_id):
        """
        Init iTag client.
        """

        # init iTag sdk
        self._itag = ItagSdk(
            config=open_api_models.Config(
                tenant_id, token, endpoint="itag2.alibaba-inc.com"
            ),
            buc_no=employee_id
        )

        # init AlphaD
        self._alphad = AlphaDataSdk(
            config=open_api_models.Config(
                tenant_id, token, endpoint="alphad.alibaba-inc.com"
            ),
            buc_no=employee_id
        )

    def _task_result_parser(self):
        """
        Parse task result.
        """
        pass

    def create_dataset(self, dataset_file_path):
        """
        Create a dataset from local file.
        """

        # Create dataset from local file
        with open(dataset_file_path, 'rb') as f:
            create_dataset_response = self._alphad.create_dataset(tenant_id, alphad_model.CreateDatasetRequest(
                data_source="LOCAL_FILE",
                dataset_name="llm_evals_datasets_rank",
                owner_name="班扬",
                owner_employee_id=self._employee_id,
                file_name="llm_evals_datasets_rank.csv",
                file=f,
                content_type="multipart/form-data",
                secure_level=1,
                remark="test dataset"
            ))
        logger.info(f'>>create dataset resp: {create_dataset_response}')

        # Get created dataset id
        dataset_id = create_dataset_response.body.result
        logger.info(f'>>create dataset id: {dataset_id}')

        # To wait for dataset creation
        while True:
            dataset_response = self._alphad.get_dataset(tenant_id, dataset_id)
            status = dataset_response.body.result.status
            if not status:
                raise ValueError("dataset status error")

            if status == "FINISHED":
                break

            time.sleep(5)

    def get_dataset_list(self):
        """
        Get dataset list on the iTag.
        """
        datasets_list_resp = self._alphad.list_datasets(self._tenant_id, alphad_model.ListDatasetsRequest(
            page_size=10,
            page_num=1,
            contain_deleted=False,
            source="_itag",
            set_type="ALPHAD_TABLE",
            shared_type="USABLE",
            creator_id=self._employee_id
        ))

        return datasets_list_resp

    def get_dataset_info(self, dataset_id: Union[str, int]):
        """
        Get dataset info.
        """
        dataset_id = str(dataset_id)
        dataset_info = self._alphad.get_dataset(self._tenant_id, dataset_id)

        return dataset_info

    def create_tag_task(self, task_name: str, dataset_id: Union[str, int], template_id: str):
        """
        Create a iTag task.
        """
        dataset_info = self.get_dataset_info(dataset_id)
        status = dataset_info.body.result.status
        logger.info(f'Current status of dataset: {status}')

        create_task_request = open_itag_models.CreateTaskRequest(
            body=open_itag_models.CreateTaskDetail(
                task_name=task_name,
                template_id=template_id,
                task_workflow=[
                    open_itag_models.CreateTaskDetailTaskWorkflow(
                        node_name='MARK'
                    )
                ],
                admins=open_itag_models.CreateTaskDetailAdmins(),
                assign_config=open_itag_models.TaskAssginConfig(
                    assign_count=1,
                    assign_type='FIXED_SIZE'
                ),
                uuid=str(uuid.uuid4()),
                task_template_config=open_itag_models.TaskTemplateConfig(),
                dataset_proxy_relations=[
                    open_itag_models.DatasetProxyConfig(
                        # source_dataset_id=dataset_info.get("result"),
                        source_dataset_id=str(dataset_id),
                        dataset_type='LABEL',
                        source='ALPHAD'
                    )
                ]
            )
        )
        create_task_response = self._itag.create_task(tenant_id, create_task_request)

        return create_task_response

    def get_tag_task_result(self):
        """
        Fetch tag task result.
        """
        pass


if __name__ == "__main__":

    tenant_id = '268ef75a'
    employee_id = '147543'

    token = """-----BEGIN RSA PRIVATE KEY-----
xxxx
xxx
xxx
-----END RSA PRIVATE KEY-----
    """

    itag_manager = ItagManager(tenant_id=tenant_id, token=token, employee_id=employee_id)

    # dataset_path = os.path.join(os.path.dirname('__file__'), 'datasets/llm_evals_datasets_rank.csv')
    # itag_manager.create_dataset(dataset_path)

    # dataset_list = itag_manager.get_dataset_list()
    # print(dataset_list)

    dataset_info = itag_manager.get_dataset_info(dataset_id=329240)
    print(dataset_info)

    # template name: xc_response_rank_0329
    # template_id = '1640901365857796096'
    # task_name = 'task_test_llm_evals_rank'
    # dataset_id = 329240
    #
    # create_task_response = itag_manager.create_tag_task(task_name=task_name,
    #                                                     dataset_id=dataset_id,
    #                                                     template_id=template_id)
    # print('>>create_task_response: ', create_task_response)
