// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _PARSE_COMMAND_LINE
#define _PARSE_COMMAND_LINE

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdlib.h>

class CommandLine {
public:
  int argc;
  char** argv;

public:
  CommandLine(int _argc, char ** _argv) : argc(_argc), argv(_argv) {}

  void BadArgument() {
    std::cout << "usage: " << argv[0] << " bad argument" << std::endl;
    abort();
  }

  char* GetOptionValue(std::string option) {
    for (int i = 1; i < argc; i++)
      if ((std::string) argv[i] == option) return argv[i+1];
    return NULL;
  }

  std::string GetOptionValue(std::string option, std::string defaultValue) {
    for (int i = 1; i < argc; i++)
      if ((std::string) argv[i] == option) return (std::string) argv[i+1];
    return defaultValue;
  }

  int GetOptionIntValue(std::string option, int defaultValue) {
    for (int i = 1; i < argc; i++)
      if ((std::string) argv[i] == option) {
	int r = atoi(argv[i+1]);
	return r;
      }
    return defaultValue;
  }

  long GetOptionLongValue(std::string option, long defaultValue) {
    for (int i = 1; i < argc; i++)
      if ((std::string) argv[i] == option) {
	long r = atol(argv[i+1]);
	return r;
      }
    return defaultValue;
  }

  double GetOptionDoubleValue(std::string option, double defaultValue) {
    for (int i = 1; i < argc; i++)
      if ((std::string) argv[i] == option) {
	double val;
	if (sscanf(argv[i+1], "%lf",  &val) == EOF) {
	  BadArgument();
	}
	return val;
      }
    return defaultValue;
  }

};
 
#endif // _PARSE_COMMAND_LINE
