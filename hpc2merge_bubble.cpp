#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

class MergeSort
{
private:
    vector<int> data;

    void merge(int start1, int end1, int start2, int end2)
    {
        vector<int> temp;

        int i = start1, j = start2;

        while (i <= end1 && j <= end2)
        {
            if (data[i] < data[j])
            {
                temp.push_back(data[i]);
                i++;
            }
            else
            {
                temp.push_back(data[j]);
                j++;
            }
        }
        while (i <= end1)
        {
            temp.push_back(data[i]);
            i++;
        }
        while (j <= end2)
        {
            temp.push_back(data[j]);
            j++;
        }
        for (int k = 0; k < temp.size(); k++)
        {
            data[start1 + k] = temp[k];
        }
    }
    void mergesort(int i, int j)
    {
        if (i < j)
        {
            int mid = (i + j) / 2;

#pragma omp parallel sections
            {
#pragma omp section
                {
                    mergesort(i, mid);
                }
#pragma omp section
                {
                    mergesort(mid + 1, j);
                }
            }

            merge(i, mid, mid + 1, j);
        }
    }

public:
    MergeSort(const vector<int> &input)
    {
        data = input;
    }

    void sequentialSort()
    {
        mergesortSequential(0, data.size() - 1);
    }

    void mergesortSequential(int i, int j)
    {
        if (i < j)
        {
            int mid = (i + j) / 2;
            mergesortSequential(i, mid);
            mergesortSequential(mid + 1, j);
            merge(i, mid, mid + 1, j);
        }
    }

    void parallelSort()
    {
#pragma omp parallel
        {
#pragma omp single
            {
                mergesort(0, data.size() - 1);
            }
        }
    }

    void print() const
    {
        for (int num : data)
        {
            cout << num << " ";
        }
        cout << endl;
    }

    void reset(const vector<int> &input)
    {
        data = input;
    }
};

class BubbleSort
{
private:
    vector<int> data;

    void swap(int &a, int &b)
    {
        int temp = a;
        a = b;
        b = temp;
    }

public:
    BubbleSort(const vector<int> &input)
    {
        data = input;
    }

    void sequentialSort()
    {
        int n = data.size();
        bool swapped;
        for (int i = 0; i < n; i++)
        {
            swapped = false;
            for (int j = 0; j < n - 1; j++)
            {
                if (data[j] > data[j + 1])
                {
                    swap(data[j], data[j + 1]);
                    swapped = true;
                }
            }
            if (!swapped)
                break;
        }
    }

    void parallelSort()
    {
        int n = data.size();
        bool swapped;
        for (int i = 0; i < n; i++)
        {
            swapped = false;
            int first = i % 2;
            #pragma omp parallel for shared(first) reduction(| : swapped)
            for (int j = first; j < n - 1; j += 2)
            {
                if (data[j] > data[j + 1])
                {
                    swap(data[j], data[j + 1]);
                    swapped = true;
                }
            }
            if (!swapped)
                break;
        }
    }

    void print() const
    {
        for (int num : data)
        {
            cout << num << " ";
        }
        cout << endl;
    }

    void reset(const vector<int> &input)
    {
        data = input;
    }
};

int main()
{
    int n = 100000; // 10 million elements
    vector<int> input(n);
    for (int i = 0; i < n; i++)
    {
        input[i] = rand() % 100000; // Random numbers between 0 and 100000
    }

    // BUBBLE SORT
    BubbleSort bubbleSorter(input);

    double start_time = omp_get_wtime();
    bubbleSorter.sequentialSort();
    double end_time = omp_get_wtime();
    cout << "\nBubble Sort - Sequential:" << endl;
    // bubbleSorter.print();
    cout << "Time taken: " << end_time - start_time << " seconds" << endl;

    cout<<input[n-1]<<endl;
    bubbleSorter.reset(input);

    start_time = omp_get_wtime();
    bubbleSorter.parallelSort();
    end_time = omp_get_wtime();
    cout << "\nBubble Sort - Parallel:" << endl;
    // bubbleSorter.print();
    cout << "Time taken: " << end_time - start_time << " seconds" << endl;

    // MERGE SORT
    MergeSort mergeSorter(input);

    start_time = omp_get_wtime();
    mergeSorter.sequentialSort();
    end_time = omp_get_wtime();
    cout << "\nMerge Sort - Sequential:" << endl;
    // mergeSorter.print();
    cout << "Time taken: " << end_time - start_time << " seconds" << endl;

    mergeSorter.reset(input);

    start_time = omp_get_wtime();
    mergeSorter.parallelSort();
    end_time = omp_get_wtime();
    cout << "\nMerge Sort - Parallel:" << endl;
    // mergeSorter.print();
    cout << "Time taken: " << end_time - start_time << " seconds" << endl;

    return 0;
}
