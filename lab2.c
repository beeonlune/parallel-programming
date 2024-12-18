#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

// Основные функции
void boundaryValueProblem(double *result, int points, double rightBoundary, int numThreads);
void buildMatrix(double **matrix, double *values, int points, double step, int numThreads);
void solveSystem(double **matrix, double *corrections, int size, int numThreads);
void logExecutionTime(const char *filename, double timeElapsed);

// Основной расчет
void boundaryValueProblem(double *result, int points, double rightBoundary, int numThreads) {
    double step = rightBoundary / (points - 1);

    // Инициализация массива решений
    result[0] = 0.0;
    result[points - 1] = rightBoundary; // Значение b, которое зададим в качества аргумента
    for (int i = 1; i < points - 1; i++) {
        result[i] = result[0] + i * step;
    }

    /* 
    Матрица для внутренней части сетки:
    - в каждом шаге вычисляется матрица с использованием текущего приближения решения
    - вычисляются поправки corrections, используя численный метод Нумерова
    - решается СЛАУ для поправок методом Гаусса
    */
    double **matrix = (double **)malloc((points - 2) * sizeof(double *));
    for (int i = 0; i < points - 2; i++) {
        matrix[i] = (double *)malloc((points - 1) * sizeof(double));
    }

    // Итерационные улучшения решения
    for (int iter = 0; iter < 3; iter++) {
        buildMatrix(matrix, result, points, step, numThreads); // Строится матрица для метода Нумерова

        double *corrections = (double *)malloc((points - 2) * sizeof(double));
        for (int i = 0; i < points - 2; i++) {
            corrections[i] = -(result[i + 2] - 2 * result[i + 1] + result[i] - step * step / 12.0 * 
                                (exp(result[i + 2]) + 10 * exp(result[i + 1]) + exp(result[i])));
        }

        for (int i = 0; i < points - 2; i++) {
            matrix[i][points - 2] = corrections[i];
        }

        solveSystem(matrix, corrections, points - 2, numThreads);

        for (int i = 0; i < points - 2; i++) {
            result[i + 1] += corrections[i];
        }

        free(corrections);
    }

    for (int i = 0; i < points - 2; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Построение матрицы для метода Нумерова
void buildMatrix(double **matrix, double *values, int points, double step, int numThreads) {
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < points - 2; i++) {
        for (int j = 0; j < points - 2; j++) {
            matrix[i][j] = 0.0;
        }
    }

    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < points - 2; i++) {
        matrix[i][i] = -2.0 - (5.0 * step * step * exp(values[i + 1]) / 6.0); // Главная диагональ
    }

    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < points - 3; i++) {
        matrix[i][i + 1] = 1.0 - (step * step / 12.0) * exp(values[i + 2]); // Верхняя диагональ
    }

    #pragma omp parallel for num_threads(numThreads)
    for (int i = 1; i < points - 2; i++) {
        matrix[i][i - 1] = 1.0 - (step * step / 12.0) * exp(values[i]); // Нижняя диагональ
    }
}

// Решение СЛАУ методом Гаусса с параллельными элементами
void solveSystem(double **matrix, double *corrections, int size, int numThreads) {
    // Прямой ход 
    for (int k = 0; k < size - 1; k++) { 
        #pragma omp parallel for num_threads(numThreads)
        for (int i = k + 1; i < size; i++) {
            double factor = matrix[i][k] / matrix[k][k];

            for (int j = k; j < size + 1; j++) {
                matrix[i][j] -= factor * matrix[k][j];
            }
        }
    }

    // Обратный ход
    for (int i = size - 1; i >= 0; i--) {
        corrections[i] = matrix[i][size] / matrix[i][i]; // Массив решений

        #pragma omp parallel for num_threads(numThreads)
        for (int j = 0; j < i; j++) {
            matrix[j][size] -= matrix[j][i] * corrections[i];
        }
    }
}

// Логирование времени выполнения
void logExecutionTime(const char *filename, double timeElapsed) {
    FILE *file = fopen(filename, "a");
    if (file != NULL) {
        fprintf(file, "%.6f\n", timeElapsed);
        fclose(file);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <points> <rightBoundary> <threads>\n", argv[0]);
        return 1;
    }

    int points = atoi(argv[1]);
    double rightBoundary = atof(argv[2]);
    int numThreads = atoi(argv[3]);

    double *result = (double *)malloc(points * sizeof(double));

    double startTime = omp_get_wtime();
    boundaryValueProblem(result, points, rightBoundary, numThreads);
    double endTime = omp_get_wtime();

    double executionTime = endTime - startTime;
    printf("Execution time: %.6f seconds\n", executionTime);

    if (numThreads == 1) {
        logExecutionTime("serial_times.log", executionTime);
    } else {
        logExecutionTime("parallel_times.log", executionTime);
    }

    FILE *outputFile = fopen("solution_output.txt", "w");
    if (outputFile != NULL) {
        for (int i = 0; i < points; i++) {
            fprintf(outputFile, "%.6f %.6f\n", i * rightBoundary / (points - 1), result[i]);
        }
        fclose(outputFile);
    }

    free(result);
    return 0;
}
