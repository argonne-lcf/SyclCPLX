SRC = $(wildcard *.cpp)
.PHONY: exe
exe: $(SRC:%.cpp=%.exe)

pEXE = $(wildcard *.exe)
.PHONY: run
run: $(addprefix run_, $(basename $(pEXE)))

%.exe: %.cpp
	-$(CXX) $(CXXFLAGS) $(CURDIR)/$< -o $(CURDIR)/$@


run_%: %.exe
	-$(TIMEOUT) $(CURDIR)/$<

.PHONY: clean
clean:
	rm -f -- $(pEXE) 
