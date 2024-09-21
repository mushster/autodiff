CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lm

SRCS = main.c differentiable_operation.c operations.c graph_utils.c iris_data.c
OBJS = $(SRCS:.c=.o)
DEPS = differentiable_operation.h operations.h graph_utils.h iris_data.h
EXEC = iris_softmax_regression

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)