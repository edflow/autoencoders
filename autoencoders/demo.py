import torch
import streamlit as st
from edflow.util.edexplore import isimage, st_get_list_or_dict_item

from autoencoders.models.bigae import BigAE


@st.cache(allow_output_mutation=True)
def get_state(gpu, name="animals"):
    model = BigAE.from_pretrained(name)
    if gpu:
        model.cuda()
    state = {"model": model}
    return state


def reconstruction(ex, config):
    st.write("Options")

    name = st.selectbox(
        "Model Name",
        ["animals", "animalfaces"],
    )

    if torch.cuda.is_available():
        gpu = st.checkbox("gpu", value=True)

    state = get_state(name=name, gpu=gpu)
    model = state["model"]

    image, image_key = st_get_list_or_dict_item(ex, "image",
                                                description="input image",
                                                filter_fn=isimage)
    st.write("Input")
    st.image((image+1.0)/2.0)

    xin = torch.tensor(image)[None,...].transpose(3,2).transpose(2,1).float()
    if gpu:
        xin = xin.cuda()

    p = model.encode(xin)
    xout = model.decode(p.mode())

    xout = xout[0].transpose(0,1).transpose(1,2)
    xout = xout.detach().cpu().numpy()
    st.write("Reconstruction")
    st.image((xout+1.0)/2.0)
