#include <rapids_logger/logger.hpp>

#include <sstream>

int main()
{
  std::ostringstream output;
  rapids_logger::logger logger("relocation", output);
  logger.info("relocation %d", 42);
  logger.flush();
  return output.str().find("relocation 42") == std::string::npos ? 1 : 0;
}
