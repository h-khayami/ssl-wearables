import json
import numpy as np
from collections import Counter

class DatasetLabelMapper:
    def __init__(self, mapping_file):
        with open(mapping_file, 'r') as f:
            self._mapping_data = json.load(f)
        self.reference_activities = self._mapping_data["reference_activities"]
        self.reverse_mapping_ref = {v: k for k, v in self.reference_activities.items()}
        self._dataset_mappings = self._mapping_data["dataset_mappings"]

    def get_mapping(self, dataset1, dataset2="reference"):
        mapping1 = self._dataset_mappings.get(dataset1, {})
        if dataset2 == "reference":
            reverse_mapping2 = self.reverse_mapping_ref
        else:
            mapping2 = self._dataset_mappings.get(dataset2, {})
            reverse_mapping2 = {v: k for k, v in mapping2.items()}
        return {label: reverse_mapping2.get(mapping1.get(label, None), None) for label in mapping1}

    def map_labels_to_numeric(self, y_test, dataset_name="reference"):
        if dataset_name == "reference":
            return np.vectorize(lambda x: self.reference_activities.get(x, -1))(y_test)
        else: 
            dataset_mapping = self._dataset_mappings.get(dataset_name, {})
            labels_map = {k.lower(): v for k, v in dataset_mapping.items()} #convert keys to lower case
            mapper = np.vectorize(lambda x: labels_map.get(x.lower(), -1)) #convert labels to lower case and then map
            return mapper(y_test)
    
    def map_numeric_to_labels(self, y_test, dataset_name="reference"):
        if dataset_name == "reference":
            return np.vectorize(lambda x: self.reverse_mapping_ref.get(x, None))(y_test)
        else:
            mapping = self._dataset_mappings.get(dataset_name, {})
            reverse_mapping = {v: k for k, v in mapping.items()}
            return np.vectorize(lambda x: reverse_mapping.get(x, None))(y_test)

    def get_unique_labels(self, y_test, dataset_name="reference"):
        return self.map_numeric_to_labels(np.unique(y_test), dataset_name)

    def get_label_distribution(self, y_test, dataset_name="reference"):
        unique, counts = np.unique(y_test, return_counts=True)
        unique_labels = self.map_numeric_to_labels(unique, dataset_name)
        return dict(zip(unique_labels, counts))
    
    def get_label_mapping(self, dataset_name):
        return self._dataset_mappings.get(dataset_name, {})

# Example Usage
# mapper = DatasetLabelMapper("activity_label_mapping.json")
# y_test_mapped = mapper.map_labels_to_numeric("WISDM", np.array(["Walking", "Jogging", "Sitting"]))
def test():
    # Example Usage
    dataset_mapping_path = "conf/cross_dataset_mapping/Activity_label_mapping.json"
    mapper = DatasetLabelMapper(dataset_mapping_path)
    y_test_mapped = mapper.map_labels_to_numeric("WISDM", np.array(["Walking", "Jogging", "Sitting", "walking"]))
    print(y_test_mapped)
    y_test_mapped = mapper.map_labels_to_numeric("MyMove", np.array(["upright_standing", "cycling", "sedentary_sitting_other", "upright_standing", "cycling", "sedentary_sitting_other"]))
    print(y_test_mapped)
    print("Unique labels of y_test from reference:",mapper.get_unique_labels(y_test_mapped))
    print("Unique labels of y_test from MyMove:",mapper.get_unique_labels(y_test_mapped, "MyMove"))
    print("label distribution of y_test:",mapper.get_label_distribution( y_test_mapped))
    print("WISDM to MyMove mapping:",mapper.get_mapping("WISDM", "MyMove"))
    print("MyMove to WISDM mapping:",mapper.get_mapping("MyMove", "WISDM"))
    print("MyMove to ExtraSensory mapping:",mapper.get_mapping("MyMove", "ExtraSensory"))
    print("ExtraSensory to MyMove mapping:",mapper.get_mapping("ExtraSensory", "MyMove"))
    print("MyMove to reference mapping:",mapper.get_mapping("MyMove"))
    # print(mapper.dataset_mappings.get("WISDM", {}))
    # print(mapper.dataset_mappings["WISDM"].get("Jogging", None))
if __name__ == "__main__":
    test()
