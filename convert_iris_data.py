def convert_iris_class(class_name):
    if class_name == "Iris-setosa":
        return 0
    elif class_name == "Iris-versicolor":
        return 1
    elif class_name == "Iris-virginica":
        return 2
    else:
        raise ValueError(f"Unknown class: {class_name}")

with open('iris.data', 'r') as f, open('iris_data.c', 'w') as out:
    out.write('#include "iris_data.h"\n\n')
    out.write('IrisData iris_dataset[IRIS_SAMPLES] = {\n')
    
    for line in f:
        if line.strip():
            features, class_name = line.strip().rsplit(',', 1)
            features = [float(x) for x in features.split(',')]
            class_index = convert_iris_class(class_name)
            out.write(f'    {{{{{features[0]:.1f}, {features[1]:.1f}, {features[2]:.1f}, {features[3]:.1f}}}, {class_index}}},\n')
    
    out.write('};\n')