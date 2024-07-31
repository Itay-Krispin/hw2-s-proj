#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Define structures */
struct cord {
    double value;
    struct cord *next;
};

struct vector {
    struct vector *next;
    struct cord *cords;
};

/* Declare functions */
void check_inputs(int argc, char *argv[]);
void print_vectors(struct vector *v);
struct vector* create_vectors(void);
struct vector* create_vectors_from_pyobject(PyObject *data);
struct vector* copy_first_k_vectors(struct vector *head_vec, int k);
struct cord* copy_cords(struct cord *head_cord);
double euclidean_distance(struct cord *cords1, struct cord *cords2);
void assign_clusters(struct vector *head_vec, struct vector *centroids, struct vector **clusters, int k);
struct vector* calculate_centroids(struct vector **clusters, int k);
int centroids_converged(struct vector *old_centroids, struct vector *new_centroids, double epsilon);
void free_vectors(struct vector *head_vec);
void free_cords(struct cord *head_cord);
PyObject* vector_to_pyobject(struct vector *head_vec);
void delete_last_vector(struct vector *head_vec);
int get_cord_length(struct cord *head_cord);
struct cord* init_empty_cord(int length);
void add_one_cord_values_to_another_cord(struct cord *add_from, struct cord *add_to);
void divide_cord_values_by_number(struct cord *head_cord, int number);

/* K-means function */
static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject *initial_centroids_py, *data_py;
    int k, iter, i;
    double epsilon;
    struct vector *data, *centroids, *new_centroids;
    struct vector **clusters;

    if (!PyArg_ParseTuple(args, "OOiid", &initial_centroids_py, &data_py, &k, &iter, &epsilon)) {
        return NULL;
    }

    data = create_vectors_from_pyobject(data_py);
    centroids = create_vectors_from_pyobject(initial_centroids_py);

    clusters = malloc(k * sizeof(struct vector*));
    if (clusters == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for clusters.");
        free_vectors(data);
        free_vectors(centroids);
        return NULL;
    }

    for (i = 0; i < k; i++) {
        clusters[i] = NULL;
    }

    for (i = 0; i < iter; i++) {
        assign_clusters(data, centroids, clusters, k);
        new_centroids = calculate_centroids(clusters, k);

        if (centroids_converged(centroids, new_centroids, epsilon)) {
            free_vectors(centroids);
            centroids = new_centroids;
            break;
        }

        free_vectors(centroids);
        centroids = new_centroids;
    }

    PyObject *result = vector_to_pyobject(centroids);

    free_vectors(data);
    free_vectors(centroids);
    for (i = 0; i < k; i++) {
        if (clusters[i] != NULL) {
            free_vectors(clusters[i]);
        }
    }
    free(clusters);

    return result;
}

/* Converts Python list to C vector linked list */
struct vector* create_vectors_from_pyobject(PyObject *data) {
    int i, j, num_vectors, num_cords;
    PyObject *vector_py, *cord_py;
    struct vector *head_vec, *curr_vec, *new_vec;
    struct cord *head_cord, *curr_cord, *new_cord;

    num_vectors = PyList_Size(data);
    // printf("num_vectors = %d\n", num_vectors);

    head_vec = malloc(sizeof(struct vector));
    if (head_vec == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for head_vec.");
        return NULL;
    }
    curr_vec = head_vec;
    curr_vec->next = NULL;

    for (i = 0; i < num_vectors; i++) {
        vector_py = PyList_GetItem(data, i);
        num_cords = PyList_Size(vector_py);

        head_cord = malloc(sizeof(struct cord));
        if (head_cord == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for head_cord.");
            free_vectors(head_vec);
            return NULL;
        }
        curr_cord = head_cord;
        curr_cord->next = NULL;

        for (j = 0; j < num_cords; j++) {
            cord_py = PyList_GetItem(vector_py, j);
            curr_cord->value = PyFloat_AsDouble(cord_py);

            if (j < num_cords - 1) {
                new_cord = malloc(sizeof(struct cord));
                if (new_cord == NULL) {
                    PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for new_cord.");
                    free_cords(head_cord);
                    free_vectors(head_vec);
                    return NULL;
                }
                curr_cord->next = new_cord;
                curr_cord = new_cord;
                curr_cord->next = NULL;
            }
        }

        curr_vec->cords = head_cord;

        if (i < num_vectors - 1) {
            new_vec = malloc(sizeof(struct vector));
            if (new_vec == NULL) {
                PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for new_vec.");
                free_vectors(head_vec);
                return NULL;
            }
            curr_vec->next = new_vec;
            curr_vec = new_vec;
            curr_vec->next = NULL;
        }
    }

    return head_vec;
}

/* Converts C vector linked list to Python list */
PyObject* vector_to_pyobject(struct vector *head_vec) {
    PyObject *result_list, *vector_list, *cord_value;
    struct vector *curr_vec;
    struct cord *curr_cord;

    result_list = PyList_New(0);
    if (result_list == NULL) {
        return NULL;
    }

    curr_vec = head_vec;
    while (curr_vec != NULL) {
        vector_list = PyList_New(0);
        if (vector_list == NULL) {
            Py_DECREF(result_list);
            return NULL;
        }

        curr_cord = curr_vec->cords;
        while (curr_cord != NULL) {
            cord_value = PyFloat_FromDouble(curr_cord->value);
            if (cord_value == NULL) {
                Py_DECREF(vector_list);
                Py_DECREF(result_list);
                return NULL;
            }
            PyList_Append(vector_list, cord_value);
            Py_DECREF(cord_value);
            curr_cord = curr_cord->next;
        }

        PyList_Append(result_list, vector_list);
        Py_DECREF(vector_list);
        curr_vec = curr_vec->next;
    }

    return result_list;
}

/* Python method definitions */
static PyMethodDef MyKMeansSPMethods[] = {
    {"fit", fit, METH_VARARGS, "Run K-means clustering algorithm.\n\nArguments:\ninitial_centroids: List of initial centroids\n data: List of data points\nk: Number of clusters\niter: Maximum number of iterations\n"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef mykmeansspmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "K-means clustering module",
    -1,
    MyKMeansSPMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&mykmeansspmodule);
}

/* Check input validity */
void check_inputs(int argc, char *argv[]) {
    double k, iter;
    char *endptr;

    /* Check if the correct number of arguments are provided */
    if (argc != 3) {
        printf("Please insert 3 arguments only: k, iter, input_file\n");
        exit(1);
    }

    /* Validate and convert the first argument to a double */
    k = strtod(argv[1], &endptr);
    if (*endptr != '\0' || k <= 1 || floor(k) != k) {
        printf("Invalid number of clusters!");
        exit(1);
    }

    /* Validate and convert the second argument to a double */
    iter = strtod(argv[2], &endptr);
    if (*endptr != '\0' || iter <= 1 || iter >= 1000 || floor(iter) != iter) {
        printf("Invalid maximum iteration!");
        exit(1);
    }
}

struct vector* create_vectors(void) {
    struct vector *head_vec, *curr_vec, *new_vec;
    struct cord *head_cord, *curr_cord;
    double n;
    char c;

    head_vec = malloc(sizeof(struct vector));
    if (head_vec == NULL) {
        printf("Memory allocation failed for head_vec.\n");
        exit(1);
    }
    curr_vec = head_vec;
    curr_vec->next = NULL;

    head_cord = malloc(sizeof(struct cord));
    if (head_cord == NULL) {
        printf("Memory allocation failed for head_cord.\n");
        exit(1);
    }
    curr_cord = head_cord;
    curr_cord->next = NULL;

    while (scanf("%lf%c", &n, &c) == 2) {
        if (c == '\n') {
            curr_cord->value = n;
            curr_vec->cords = head_cord;

            new_vec = malloc(sizeof(struct vector));
            if (new_vec == NULL) {
                printf("Memory allocation failed for new_vec.\n");
                exit(1);
            }
            curr_vec->next = new_vec;
            curr_vec = new_vec;
            curr_vec->next = NULL;

            head_cord = malloc(sizeof(struct cord));
            if (head_cord == NULL) {
                printf("Memory allocation failed for head_cord.\n");
                exit(1);
            }
            curr_cord = head_cord;
            curr_cord->next = NULL;
            continue;
        }

        curr_cord->value = n;
        curr_cord->next = malloc(sizeof(struct cord));
        if (curr_cord->next == NULL) {
            printf("Memory allocation failed for curr_cord->next.\n");
            exit(1);
        }
        curr_cord = curr_cord->next;
        curr_cord->next = NULL;
    }

    /* Free the last allocated cord which is not used */
    free(curr_cord);

    /* Delete the last vector which is extra */
    delete_last_vector(head_vec);

    return head_vec;
}

struct cord* copy_cords(struct cord *old_cord_head) {
    struct cord *old_cord_curr, *new_cord_curr, *new_cord_head;

    old_cord_curr = old_cord_head;
    new_cord_head = malloc(sizeof(struct cord));
    if (new_cord_head == NULL) {
        printf("Memory allocation failed for new_cord_head.\n");
        exit(1);
    }
    new_cord_curr = new_cord_head;

    while (old_cord_curr != NULL) {
        new_cord_curr->value = old_cord_curr->value;
        old_cord_curr = old_cord_curr->next;
        if (old_cord_curr != NULL) {
            new_cord_curr->next = malloc(sizeof(struct cord));
            if (new_cord_curr->next == NULL) {
                printf("Memory allocation failed for new_cord_curr->next.\n");
                free_cords(new_cord_head);
                exit(1);
            }
            new_cord_curr = new_cord_curr->next;
        } else {
            new_cord_curr->next = NULL;
        }
    }

    return new_cord_head;
}

int get_cord_length(struct cord *head_cord) {
    int count = 0;
    struct cord *curr_cord = head_cord;
    while (curr_cord != NULL) {
        count++;
        curr_cord = curr_cord->next;
    }
    return count;
}

int get_vector_length(struct vector *head_vec) {
    int count = 0;
    struct vector *curr_vec = head_vec;
    while (curr_vec != NULL) {
        count++;
        curr_vec = curr_vec->next;
    }
    return count;
}

struct cord* init_empty_cord(int length) {
    int i;
    struct cord *head_cord, *curr_cord;
    head_cord = malloc(sizeof(struct cord));
    if (head_cord == NULL) {
        printf("Memory allocation failed for head_cord.\n");
        exit(1);
    }
    curr_cord = head_cord;
    curr_cord->next = NULL;

    for (i = 0; i < length - 1; i++) {
        curr_cord->value = 0;
        curr_cord->next = malloc(sizeof(struct cord));
        if (curr_cord->next == NULL) {
            printf("Memory allocation failed for curr_cord->next.\n");
            free_cords(head_cord);
            exit(1);
        }
        curr_cord = curr_cord->next;
        curr_cord->next = NULL;
    }
    return head_cord;
}

void add_one_cord_values_to_another_cord(struct cord *add_from, struct cord *add_to) {
    struct cord *curr_cord_from, *curr_cord_to;
    curr_cord_from = add_from;
    curr_cord_to = add_to;

    while (curr_cord_from != NULL && curr_cord_to != NULL) {
        curr_cord_to->value += curr_cord_from->value;
        curr_cord_from = curr_cord_from->next;
        curr_cord_to = curr_cord_to->next;
    }
}

void divide_cord_values_by_number(struct cord *head_cord, int number) {
    struct cord *curr_cord;
    curr_cord = head_cord;

    while (curr_cord != NULL) {
        curr_cord->value /= number;
        curr_cord = curr_cord->next;
    }
}

struct vector* copy_first_k_vectors(struct vector *old_vec_head, int k) {
    struct vector *old_vec_curr, *new_vec_head, *new_vec_curr;
    int i;

    old_vec_curr = old_vec_head;
    new_vec_curr = malloc(sizeof(struct vector));
    if (new_vec_curr == NULL) {
        printf("Memory allocation failed for new_vec_curr.\n");
        exit(1);
    }
    new_vec_head = new_vec_curr;

    for (i = 0; i < k; i++) {
        new_vec_curr->cords = copy_cords(old_vec_curr->cords);
        if (i == k - 1) {
            new_vec_curr->next = NULL;
        } else {
            new_vec_curr->next = malloc(sizeof(struct vector));
            if (new_vec_curr->next == NULL) {
                printf("Memory allocation failed for new_vec_curr->next.\n");
                free_vectors(new_vec_head);
                exit(1);
            }
            new_vec_curr = new_vec_curr->next;
            old_vec_curr = old_vec_curr->next;
        }
    }

    return new_vec_head;
}

double euclidean_distance(struct cord *cords1, struct cord *cords2) {
    double sum = 0.0;
    while (cords1 != NULL && cords2 != NULL) {
        sum += pow(cords1->value - cords2->value, 2);
        cords1 = cords1->next;
        cords2 = cords2->next;
    }
    return sqrt(sum);
}

void assign_clusters(struct vector *all_vectors_head, struct vector *centroids_head, struct vector **clusters, int k) {
    struct vector *all_vectors_curr, *curr_centroid, *new_vec;
    int i, min_index;
    double min_dist, dist;

    /* delete old clusters */
    for (i = 0; i < k; i++) {
        if (clusters[i] != NULL) {
            free_vectors(clusters[i]);
        }
        clusters[i] = NULL;
    }

    all_vectors_curr = all_vectors_head;
    /* Run over all vectors */
    while (all_vectors_curr != NULL) {
        curr_centroid = centroids_head;
        min_dist = -1;
        min_index = 0;

        for (i = 0; i < k; i++) {
            dist = euclidean_distance(all_vectors_curr->cords, curr_centroid->cords);
            if (dist < min_dist || min_dist == -1) {
                min_dist = dist;
                min_index = i;
            }
            curr_centroid = curr_centroid->next;
        }

        /* Add the vector to the correct cluster */
        new_vec = malloc(sizeof(struct vector));
        if (new_vec == NULL) {
            printf("Memory allocation failed for new_vec.\n");
            exit(1);
        }
        new_vec->cords = copy_cords(all_vectors_curr->cords);
        new_vec->next = clusters[min_index];
        clusters[min_index] = new_vec;

        /* continue to next vector */
        all_vectors_curr = all_vectors_curr->next;
    }
}

struct vector* calculate_centroids(struct vector **clusters, int k) {
    struct vector *new_centroids_head = NULL, *new_centroids_curr = NULL, *new_vector_for_centroid, *curr_cluster;
    struct cord *new_cord_for_vector_head;
    int count, i;

    /* run over all clusters */
    for (i = 0; i < k; i++) {
        count = 0;
        new_vector_for_centroid = malloc(sizeof(struct vector));
        if (new_vector_for_centroid == NULL) {
            printf("Memory allocation failed for new_vector_for_centroid.\n");
            exit(1);
        }
        new_vector_for_centroid->next = NULL;

        new_cord_for_vector_head = init_empty_cord(get_cord_length(clusters[i]->cords));
        
        curr_cluster = clusters[i];
        while (curr_cluster != NULL) {
            count++;
            add_one_cord_values_to_another_cord(curr_cluster->cords, new_cord_for_vector_head);
            curr_cluster = curr_cluster->next;
        }
        divide_cord_values_by_number(new_cord_for_vector_head, count);
        new_vector_for_centroid->cords = new_cord_for_vector_head;
        
        if (new_centroids_head == NULL) {
            new_centroids_head = new_vector_for_centroid;
            new_centroids_curr = new_vector_for_centroid;
        } else {
            new_centroids_curr->next = new_vector_for_centroid;
            new_centroids_curr = new_centroids_curr->next;
        }
    }

    return new_centroids_head;
}

/* Check if centroids have converged */
int centroids_converged(struct vector *old_centroids, struct vector *new_centroids, double epsilon) {
    while (old_centroids != NULL && new_centroids != NULL) {
        if (euclidean_distance(old_centroids->cords, new_centroids->cords) > epsilon) {
            return 0;
        }
        old_centroids = old_centroids->next;
        new_centroids = new_centroids->next;
    }
    return 1;
}

void delete_last_vector(struct vector *head_vec) {
    struct vector *curr_vec = head_vec;
    struct vector *prev_vec = NULL;
    while (curr_vec->next != NULL) {
        prev_vec = curr_vec;
        curr_vec = curr_vec->next;
    }
    free(curr_vec);
    if (prev_vec != NULL) {
        prev_vec->next = NULL;
    }
}

/* Free memory allocated to vectors */
void free_vectors(struct vector *head_vec) {
    struct vector *temp_vec;
    struct cord *temp_cord;

    while (head_vec != NULL) {
        temp_cord = head_vec->cords;  /* Use temp_cord to store head_vec->cords */
        free_cords(temp_cord);        /* Free cords stored in temp_cord */
        temp_vec = head_vec;
        head_vec = head_vec->next;
        free(temp_vec);
    }
}

/* Free memory allocated to cords */
void free_cords(struct cord *head_cord) {
    struct cord *temp_cord;

    while (head_cord != NULL) {
        temp_cord = head_cord;
        head_cord = head_cord->next;
        free(temp_cord);
    }
}

void print_single_vector(struct vector *v) {
    struct cord *curr_cord;

    curr_cord = v->cords;
    while (curr_cord != NULL) {
        if (curr_cord->next == NULL) {
            printf("%.4f", curr_cord->value);
        } else {
            printf("%.4f,", curr_cord->value);
        }
        curr_cord = curr_cord->next;
    }
    printf("\n");
}

/* Print all vectors */
void print_vectors(struct vector *v_head) {
    struct vector *v_curr;

    v_curr = v_head;
    while (v_curr != NULL) {
        print_single_vector(v_curr);
        v_curr = v_curr->next;
    }
}