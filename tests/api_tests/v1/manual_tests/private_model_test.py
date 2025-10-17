"""

Todo: test mp
"""
import marqo.errors
import marqo


# SET THE FOLLOWING:
acc_key = ''
sec_acc_key = ''
hf_token = ''


bucket = 'model-authentication-test'
ob = 'dummy customer/vit_b_32-quickgelu-laion400m_e31-d867053b.pt'

hf_repo_name = "Marqo/test-private"
hf_object = "dummy_model.pt"


index_name = 'index_name'


mq = marqo.Client()


def _get_base_index_settings():
    return {
        "index_defaults": {
            "treat_urls_and_pointers_as_images": True,
            "model": 'my_model2',
            "normalize_embeddings": True,
            # notice model properties aren't here. Each test has to add it
        }
    }

def _get_s3_settings():
    ix_settings = _get_base_index_settings()
    ix_settings['index_defaults']['model_properties'] = {
        "name": "ViT-B/32",
        "dimensions": 512,
        "model_location": {
            "s3": {
                "Bucket": bucket,
                "Key": ob,
            },
            "auth_required": True
        },
        "type": "open_clip",
    }
    return ix_settings


def _get_hf_settings():
    ix_settings = _get_base_index_settings()
    ix_settings['index_defaults']['model_properties'] = {
        "name": "ViT-B/32",
        "dimensions": 512,
        "model_location": {
            "hf": {
                "repo_id": hf_repo_name,
                "filename": hf_object,
            },
            "auth_required": True
        },
        "type": "open_clip",
    }
    return ix_settings


def clean_up():
    try:
        mq.delete_index(index_name=index_name)
    except marqo.errors.MarqoWebError:
        pass

def run_s3_test():
    """add docs -> search"""
    mq.create_index(
        index_name=index_name, settings_dict=_get_s3_settings(),
    )
    print(
        mq.index(index_name=index_name).add_documents(
            auto_refresh=True, documents=[{'a': 'b'}],
            model_auth={'s3': {"aws_access_key_id" : acc_key, "aws_secret_access_key": sec_acc_key}}
        )
    )
    print(
        mq.index(index_name=index_name).search(
            q="Hehehe",
            model_auth={'s3': {"aws_access_key_id" : acc_key, "aws_secret_access_key": sec_acc_key}}
        )
    )

def run_s3_test_search():
    """can search download the model? """
    mq.create_index(
        index_name=index_name, settings_dict=_get_s3_settings(),
    )
    mq.index(index_name=index_name).search(
        q="Hehehe", model_auth={'s3': {'aws_access_key_id':acc_key, 'aws_secret_access_key':sec_acc_key}}
    )


def run_hf_test():
    """add docs -> search"""
    mq.create_index(
        index_name=index_name, settings_dict=_get_hf_settings(),
    )
    ar =  mq.index(index_name=index_name).add_documents(
        auto_refresh=True, documents=[{'title': 'apples'}, {'title': 'office politik'}],
        model_auth={'hf': {'token': hf_token}}
    )
    print(ar)
    sr = mq.index(index_name=index_name).search(
        q="what is healthy fruit I can eat",
        model_auth={'hf': {'token': hf_token}}
    )
    print(sr)
    print('tensor_search.get_loaded_models()' ,mq.get_loaded_models())



clean_up()
run_s3_test()
clean_up()
