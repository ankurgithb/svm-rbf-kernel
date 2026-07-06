SVM RBF Kernel Visualization

This project demonstrates how the Radial Basis Function (RBF) kernel allows a Support Vector Machine (SVM) to classify data that cannot be separated with a straight line.

The project uses the make_circles dataset from scikit-learn and visualizes how the data can be separated after applying the RBF kernel.

What this project covers

* Creating a non-linearly separable dataset
* Visualizing the dataset in 2D
* Understanding the intuition behind the RBF transformation using a 3D plot
* Training an SVM with the RBF kernel
* Visualizing the resulting decision boundary

Technologies used

* Python
* NumPy
* Matplotlib
* Scikit-learn

Project structure

.
├── main.py
├── circles_dataset.png
├── circles_3d.png
├── svm_decision_boundary.png
└── README.md


Output

The following images are generated after running the script:

* circles_dataset.png – Original dataset
* circles_3d.png – 3D visualization of the RBF-inspired transformation
* svm_decision_boundary.png – Decision boundary learned by the SVM

What I learned

While building this project, I explored:

* Why a linear SVM cannot separate circular data
* How the RBF kernel measures similarity between points
* The intuition behind mapping data into a higher-dimensional space
* How the kernel trick allows SVMs to create non-linear decision boundaries
