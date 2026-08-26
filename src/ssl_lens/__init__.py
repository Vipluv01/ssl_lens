"""ssl_lens: self-supervised pretraining (SimSiam) and the question it's
built to answer -- how many labelled examples does a downstream task
actually need if the encoder is pretrained on unlabelled data first?

The label-efficiency curve (linear-probe / k-NN / fine-tune accuracy at
1%/10%/100% of labels) is the deliverable; SimSiam is the pretraining method
chosen to get there, over SimCLR/BYOL, for a concrete, budget-driven reason
documented in simsiam.py.
"""
