## Dataset

For pre-processed datasets, please refer to the bottom text of DATASET.md file. (Note: after downloading all the raw datasets, run the data_convert.ipynb to convert all raw data into NPY point clouds)

The overall directory structure after downloading the raw data should be:

```
│BIM-JEPA/
├──cfgs/
├──datasets/
├──data/
│   ├──IFC_extracted_elements_dataset_part_1/
│   ├──IFC_extracted_elements_dataset_part_2/
│   ├──IFC_extracted_elements_dataset_part_3/
│   ├──IFCNet/
│   ├──BIMGEOM/
│   ├──IFCNetCore/
│   ├──....../
├──......
```

**IFC_extracted_elements_dataset_part_x:** We denote this as IFC-884K in the manuscript. You can download the raw IFC-884K data from [[Zenodo]](https://zenodo.org/records/10730758). Each data part contains individual building element in OBJ format. For part 1 dataset, there are 9991 files that are named with a wrong file extension (e.g., 074508_IfcOpeningElement.ifc).obj' -- should be --> '074508_IfcOpeningElement.obj'). The directory structure looks like this:

```
│IFC_extracted_elements_dataset_part_1/
├──000001_IfcDoor.obj
├──000002_IfcOpeningElement.obj
├──......
IFC_extracted_elements_dataset_part_2/
├──300001_IfcDiscreteAccessory.obj
├──300002_IfcMember.obj
├──......
IFC_extracted_elements_dataset_part_3/
├──600001_IfcColumn.obj
├──600002_IfcColumn.obj
├──......
```

**IFCNet:** We pre-trained the model on the IFCNet data, but explicitly removed the testing split in IFCNetCore from IFCNet, as it is used in the downstream classification task. The IFCNet (given in IFC format) and IFCNetCore (given in both IFC and OBJ format) data can be downloaded at [[GitHub]](https://github.com/RWTH-E3D/ifcnet-models) or [[ifcnet.e3d.rwth-aachen.de]](https://ifcnet.e3d.rwth-aachen.de/). The directory structure looks like this:

```
│IFCNet/
├──IfcActuator/
│  ├──1949eb12c6ee48488465d0321917710b.ifc
├──IfcAirTerminal/
│  ├──0a59ac7a7fa04a9dafe8c425bab881f7.ifc
│  ├──0a59ac7a7fa04a9dafe8c425bab881f7.ifc
│  ├──......
├──.../
│  ├── ......
├──IfcWindow/
│  ├── ......
```

**IFCNetCore:** IFCNetCore is a subset of IFCNet. IFCNetCore can be similarly downloaded at [[GitHub]](https://github.com/RWTH-E3D/ifcnet-models) or [[ifcnet.e3d.rwth-aachen.de]](https://ifcnet.e3d.rwth-aachen.de/). The directory structure looks like this:

```
│IFCNetCore/
├──IfcAirTerminal/
│  ├──train/
│  │  ├──0a59ac7a7fa04a9dafe8c425bab881f7.obj/
│  │  ├──....../
│  ├──test/
│  │  ├──0b39c7a0e3fb421dbdc1d2d0a55e5b72.obj/
│  │  ├──....../
├──IfcBeam/
│  ├──train/
│  │  ├──....../
│  ├──test/
│  │  ├──....../
├──.../
```

**BIMGEOM:** You can download the dataset (in PLY format) at [[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YK86XK)]. Again, the testing split of BIMGEOM is excluded from the pre-training stage. The directory structure looks like this:

```
│BIMGEOM/
├──IfcColumn/
│  ├──train/
│  │  ├──0_IfcColumn.ply/
│  │  ├──....../
│  ├──test/
│  │  ├──982_IfcColumn.ply/
│  │  ├──....../
├──IfcDistributionControlElement/
│  ├──train/
│  │  ├──....../
│  ├──test/
│  │  ├──....../
├──.../
```

**BIMCompNet:** You can download the dataset at [[BIMCompNet](https://bimcompnet-606lab.xaut.edu.cn/)]. Used for fine-grained BIM component classification.

**BIMObject:** You can download the dataset at [[GitHub](https://github.com/duyguutkucu/BIMObjectDataset/tree/main)]. Used for BIM object classification.

**BIMNet:** You can download the dataset at [[GitHub](https://github.com/LydJason/BIMNet)]. Used for semantic segmentation via transfer learning.

**ShapeNetPart:** TBD.

---

### Pre-processed datasets

Now, after downloading all the raw datasets into the data folder, simply run the data_convert.ipynb script to pre-process all the raw datasets into point clouds (.npy format).

#### For pre-training of BIM-JEPA, it consists of all these pre-processed files:
    data/IFC_extracted_elements_dataset_part_1_pointclouds_4096
    data/IFC_extracted_elements_dataset_part_2_pointclouds_4096
    data/IFC_extracted_elements_dataset_part_3_pointclouds_4096
    data/IFCNet_NO_TEST_pointclouds_4096
    data/processed_BIMGEOM_NO_TEST_pointclouds_4096