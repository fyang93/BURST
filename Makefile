EVVE_DIR=somewhere
EVVE_FEAT_DIR=somewhere
EVVE_FEAT_DIR=somewhere
FEAT_TYPE=resnet50
FPS=5fps

.PHONY: embed
embed:
	python embed.py \
		--infos_dir $(EVVE_DIR)/dataset/$(FEAT_TYPE)_infos_$(FPS) \
		--embed_dir $(EVVE_DIR)/data/$(FEAT_TYPE)_data_$(FPS) \
		--periods 144 233 377 610 \
		--mean \
		# --pca_path ./resnet50_pca.jbl \
		# --pca \

.PHONY: retrieve
retrieve:
	python retrieve.py \
		--embed_dir $(EVVE_DIR)/data/$(FEAT_TYPE)_data_$(FPS)\
		--results_dir $(EVVE_DIR)/result/$(FEAT_TYPE)_result_$(FPS) \
		--periods 144 233 377 610 \
		--mean \

.PHONY: eval
eval:
	python combine.py \
		--annot_dir $(EVVE_DIR)/dataset/annots \
		--results_dir $(EVVE_DIR)/result/$(FEAT_TYPE)_result_$(FPS) \
		--output_path $(EVVE_DIR)/result/$(FEAT_TYPE)_result_$(FPS).dat \
		--iter 0

