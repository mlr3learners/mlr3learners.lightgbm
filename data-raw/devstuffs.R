packagename <- "mlr3learners.lightgbm"

# remove existing description object
unlink("DESCRIPTION")

# Create a new description object
my_desc <- desc::description$new("!new")

# Set your package name
my_desc$set("Package", packagename)

# Set author names 2
my_desc$set_authors(c(
  person("Lorenz A.", "Kapsner", email = "lorenz.kapsner@gmail.com", role = c("cre", "aut"),
         comment = c(ORCID = "0000-0003-1866-860X"))
))

# Remove some author fields
my_desc$del("Maintainer")

# Set the version
my_desc$set_version("0.0.3.9001")

# The title of your package
my_desc$set(Title = "mlr3: LightGBM learner")

# The description of your package
my_desc$set(Description = paste0("Brings the LightGBM functionality ",
                                 "to the mlr3 framework."))

# The date when description file has been created
my_desc$set("Date" = as.character(Sys.Date()))

# The urls
my_desc$set("URL", "https://github.com/kapsner/mlr3learners.lightgbm")
my_desc$set("BugReports", "https://github.com/kapsner/mlr3learners.lightgbm/issues")

# Vignette Builder
my_desc$set("VignetteBuilder" = "knitr")

# License
my_desc$set("License", "LGPL-3")

# Save everyting
my_desc$write(file = "DESCRIPTION")

# License
#usethis::use_lgpl_license(name = "Lorenz A. Kapsner")


# add Imports and Depends
# Listing a package in either Depends or Imports ensures that it’s installed when needed
# Imports just loads the package, Depends attaches it
# Loading will load code, data and any DLLs; register S3 and S4 methods; and run the .onLoad() function.
##      After loading, the package is available in memory, but because it’s not in the search path,
##      you won’t be able to access its components without using ::.
##      Confusingly, :: will also load a package automatically if it isn’t already loaded.
##      It’s rare to load a package explicitly, but you can do so with requireNamespace() or loadNamespace().
# Attaching puts the package in the search path. You can’t attach a package without first loading it,
##      so both library() or require() load then attach the package.
##      You can see the currently attached packages with search().

# Depends
usethis::use_package("R", min_version = "2.10", type = "Depends")

# Imports
usethis::use_package("data.table", type="Imports")
usethis::use_package("R6", type = "Imports")
usethis::use_package("paradox", type = "Imports")
usethis::use_package("mlr3misc", type = "Imports")
usethis::use_package("ggplot2", type = "Imports")
usethis::use_package("mlr3", type = "Imports")
usethis::use_package("plyr", type = "Imports")

# Suggests
usethis::use_package("testthat", type = "Suggests")
usethis::use_package("lintr", type = "Suggests")
usethis::use_package("checkmate", type = "Suggests")
# for vignettes
usethis::use_package("rmarkdown", type = "Suggests")
usethis::use_package("qpdf", type = "Suggests")
usethis::use_package("knitr", type = "Suggests")
usethis::use_package("future", type = "Suggests")
usethis::use_package("mlr3tuning", type = "Suggests")
# buildignore
usethis::use_build_ignore("LICENSE.md")
usethis::use_build_ignore(".gitlab-ci.yml")
usethis::use_build_ignore("data-raw")

# gitignore
usethis::use_git_ignore("/*")
usethis::use_git_ignore("/*/")
usethis::use_git_ignore("*.log")
usethis::use_git_ignore("!/.gitignore")
usethis::use_git_ignore("!/.gitlab-ci.yml")
usethis::use_git_ignore("!/data-raw/")
usethis::use_git_ignore("!/DESCRIPTION")
usethis::use_git_ignore("!/inst/")
usethis::use_git_ignore("!/LICENSE.md")
usethis::use_git_ignore("!/man/")
usethis::use_git_ignore("!NAMESPACE")
usethis::use_git_ignore("!/R/")
usethis::use_git_ignore("!/tests/")
usethis::use_git_ignore("!/vignettes/")
usethis::use_git_ignore("!/README.md")
usethis::use_git_ignore("!/tests/")
usethis::use_git_ignore("/.Rhistory")
usethis::use_git_ignore("!/*.Rproj")
usethis::use_git_ignore("/.Rproj*")
usethis::use_git_ignore("/.RData")

# code coverage
#covr::package_coverage()

# lint package
#lintr::lint_package()

