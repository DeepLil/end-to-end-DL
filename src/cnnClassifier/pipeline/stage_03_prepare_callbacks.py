from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier import logger


STAGE_NAME = "Prepare callback"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callback_config = config.get_prepare_callback_config
        prepare_callback = PrepareCallback(config=prepare_callback_config)
        prepare_callback.get_tb_ckpt_callbacks


if __name__ == 'main':
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<\n\nx========x")
        obj = PrepareCallback()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e