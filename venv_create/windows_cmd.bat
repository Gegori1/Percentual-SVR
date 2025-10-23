cd ..

@echo "creating virtual environment"
python -m venv env
call env\Scripts\activate

@echo "installing dependencies"
pip install -e git+https://github.com/Gegori1/gcloud_library#egg=gcp_library --config-settings editable_mode=strict
pip install -e git+https://github.com/Gegori1/Percentual-SVR-package#egg=percentual_svr --config-settings editable_mode=strict
pip install -r requirements\requirements_local.txt