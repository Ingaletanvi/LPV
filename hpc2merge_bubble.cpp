#include <iostream>
#include <omp.h>
using namespace std;

// ===== MERGE FUNCTION =====
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = new int[n1];
    int* R = new int[n2];

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

// ===== Traditional Merge Sort =====
void mergeSortTraditional(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSortTraditional(arr, left, mid);
        mergeSortTraditional(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// ===== Parallel OMP Merge Sort =====
void mergeSortOMP(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortOMP(arr, left, mid);

            #pragma omp section
            mergeSortOMP(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
    }
}

// ===== Traditional Bubble Sort =====
void bubbleSortTraditional(int arr[], int n) {
    bool swapped;

    for (int i = 0; i < n - 1; ++i) {
        swapped = false;

        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

// ===== Parallel OMP Bubble Sort =====
void bubbleSortOMP(int arr[], int n) {
    bool swapped;

    for (int i = 0; i < n - 1; ++i) {
        swapped = false;

        #pragma omp parallel for shared(arr, swapped)
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

// ===== MAIN FUNCTION =====
int main() {
    int choice, n;

    cout << "Enter number of elements: ";
    cin >> n;

    int* original = new int[n];
    cout << "Enter " << n << " elements:\n";
    for (int i = 0; i < n; ++i)
        cin >> original[i];

    cout << "\nChoose sorting algorithm:\n";
    cout << "1. Merge Sort\n";
    cout << "2. Bubble Sort\n";
    cout << "Enter choice: ";
    cin >> choice;

    double startTime, endTime;

    // Copy arrays to work separately
    int* arrTraditional = new int[n];
    int* arrOMP = new int[n];

    for (int i = 0; i < n; ++i) {
        arrTraditional[i] = original[i];
        arrOMP[i] = original[i];
    }

    switch (choice) {
        case 1:
            cout << "\n=== Traditional Merge Sort ===\n";
            startTime = omp_get_wtime();
            mergeSortTraditional(arrTraditional, 0, n - 1);
            endTime = omp_get_wtime();
            cout << "Execution Time (Traditional Merge Sort): " << (endTime - startTime) << " seconds\n";

            cout << "\n=== OMP Parallel Merge Sort ===\n";
            startTime = omp_get_wtime();
            #pragma omp parallel
            {
                #pragma omp single
                mergeSortOMP(arrOMP, 0, n - 1);
            }
            endTime = omp_get_wtime();
            cout << "Execution Time (OMP Parallel Merge Sort): " << (endTime - startTime) << " seconds\n";
            break;

        case 2:
            cout << "\n=== Traditional Bubble Sort ===\n";
            startTime = omp_get_wtime();
            bubbleSortTraditional(arrTraditional, n);
            endTime = omp_get_wtime();
            cout << "Execution Time (Traditional Bubble Sort): " << (endTime - startTime) << " seconds\n";

            cout << "\n=== OMP Parallel Bubble Sort ===\n";
            startTime = omp_get_wtime();
            bubbleSortOMP(arrOMP, n);
            endTime = omp_get_wtime();
            cout << "Execution Time (OMP Parallel Bubble Sort): " << (endTime - startTime) << " seconds\n";
            break;

        default:
            cout << "Invalid choice!\n";
            delete[] original;
            delete[] arrTraditional;
            delete[] arrOMP;
            return 1;
    }

    cout << "\n=== Final Sorted Array (OMP Version) ===\n";
    for (int i = 0; i < n; ++i)
        cout << arrOMP[i] << " ";
    cout << endl;

    delete[] original;
    delete[] arrTraditional;
    delete[] arrOMP;

    return 0;
}
